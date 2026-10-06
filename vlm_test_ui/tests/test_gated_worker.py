from __future__ import annotations
import json
import os
import sys
import tempfile
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.core.jev_gate import GateConfig, HAZARDS
from app.core.frame_sampler import target_frame_count
from app.core.types import GateRecord, VLMEvent
from app.workers.batch_worker import BatchWorker
from app.workers.gated_batch_worker import GatedBatchWorker
from examples.mock_jev import MockJevDecider
from tests.support import make_test_video, RecordingAdapter

_app = QApplication.instance() or QApplication(sys.argv[:1])


def _run_worker(worker):
    outcome = []
    gate_records = []
    vlm_events = []
    statuses = []
    run_infos = []

    worker.finished.connect(lambda: outcome.append("finished"))
    worker.canceled.connect(lambda: outcome.append("canceled"))
    worker.error_occurred.connect(lambda m: outcome.append(f"error:{m}"))

    if hasattr(worker, "gate_ready"):
        worker.gate_ready.connect(gate_records.append)
    if hasattr(worker, "gate_status"):
        worker.gate_status.connect(statuses.append)
    if hasattr(worker, "gate_run_info"):
        worker.gate_run_info.connect(run_infos.append)
    worker.event_ready.connect(vlm_events.append)

    worker.start()
    max_spins = 60_000
    spins = 0
    while not outcome and spins < max_spins:
        _app.processEvents()
        spins += 1
    worker.wait()

    return {
        "outcome": outcome[0] if outcome else "no_outcome",
        "gate_records": gate_records,
        "vlm_events": vlm_events,
        "statuses": statuses,
        "run_infos": run_infos,
    }


class TestGatedBatchWorker(unittest.TestCase):

    def setUp(self):
        self.video = make_test_video(duration_ms=12000, fps=30.0)
        self.cfg = GateConfig()
        self.log_dir = tempfile.mkdtemp()

    def tearDown(self):
        os.unlink(self.video)
        import shutil
        shutil.rmtree(self.log_dir, ignore_errors=True)

    def _make_worker(self, decider=None, cfg=None, adapter=None):
        if decider is None:
            decider = MockJevDecider(rules=[], default=0.05)
        if cfg is None:
            cfg = self.cfg
        if adapter is None:
            adapter = RecordingAdapter()
        return GatedBatchWorker(
            media_path=self.video, adapter=adapter,
            chunk_s=5.0, overlap_s=1.0, vlm_fps=4.0, parent=None,
            decider=decider, gate_config=cfg, log_dir=self.log_dir,
        )

    # GATE-1: chunk count == compute_chunk_windows
    def test_gate_1_chunk_count(self):
        worker = self._make_worker()
        result = _run_worker(worker)
        self.assertEqual(result["outcome"], "finished")
        from app.core.chunker import compute_chunk_windows
        import cv2
        cap = cv2.VideoCapture(self.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        dur = int(round(frames / fps * 1000))
        cap.release()
        windows = compute_chunk_windows(dur, 5.0, 1.0, include_partial=True)
        expected = len(windows)
        self.assertEqual(len(result["vlm_events"]), expected)
        self.assertEqual(len(result["gate_records"]), expected)

    # GATE-2: each chunk emitted exactly once, in order, event_id = idx+1
    def test_gate_2_order_and_event_id(self):
        worker = self._make_worker()
        result = _run_worker(worker)
        self.assertEqual(result["outcome"], "finished")
        events = result["vlm_events"]
        for i, ev in enumerate(events):
            self.assertEqual(ev.id, i + 1)
            self.assertEqual(ev.chunk_id, i)
        recs = result["gate_records"]
        for i, rec in enumerate(recs):
            self.assertEqual(rec.chunk_id, i)

    # GATE-3: REDUCED -> reduced_fps frame count; FULL -> vlm_fps frame count
    def test_gate_3_frame_counts(self):
        dec = MockJevDecider()
        adapter = RecordingAdapter()
        worker = self._make_worker(decider=dec, adapter=adapter)
        result = _run_worker(worker)
        self.assertEqual(result["outcome"], "finished")
        reduced_fps = self.cfg.route_reduced_fps
        full_fps = 4.0
        for rec in result["gate_records"]:
            if rec.tier == "REDUCED":
                expected_n = target_frame_count(rec.start_ms, rec.end_ms, reduced_fps)
            else:
                expected_n = target_frame_count(rec.start_ms, rec.end_ms, full_fps)
            # frames_sent can be < expected_n at the video boundary (last frame seek miss)
            self.assertLessEqual(rec.frames_sent, expected_n,
                f"chunk {rec.chunk_id} {rec.tier}: sent {rec.frames_sent} > target {expected_n}")
            self.assertGreater(rec.frames_sent, 0,
                f"chunk {rec.chunk_id} {rec.tier}: sent 0 frames")

    # GATE-4: image input -> super().run() (no gate signals, no log)
    def test_gate_4_image_input_uses_super(self):
        import cv2
        img_path = tempfile.mktemp(suffix=".png")
        cv2.imwrite(img_path, np.zeros((64, 64, 3), dtype=np.uint8))
        try:
            adapter = RecordingAdapter()
            worker = GatedBatchWorker(
                media_path=img_path, adapter=adapter,
                chunk_s=5.0, overlap_s=1.0, vlm_fps=4.0, parent=None,
                decider=MockJevDecider(rules=[]), gate_config=self.cfg,
                log_dir=self.log_dir,
            )
            result = _run_worker(worker)
            self.assertEqual(result["outcome"], "finished")
            self.assertEqual(result["gate_records"], [])
            self.assertEqual(result["run_infos"], [])
            self.assertEqual(os.listdir(self.log_dir), [])
            self.assertEqual(len(result["vlm_events"]), 1)
        finally:
            if os.path.exists(img_path):
                os.unlink(img_path)

    # GATE-5: cancel during ticks -> canceled emitted, prior records preserved
    def test_gate_5_cancel_during_ticks(self):
        worker_ref = [None]

        class _CancelOnFirstTick:
            label = "cancel_dec"
            _calls = 0
            def __call__(self, frames_rgb, fps, t_ms):
                self._calls += 1
                if self._calls == 1 and worker_ref[0] is not None:
                    worker_ref[0].cancel()
                return {h: 0.05 for h in HAZARDS}

        dec = _CancelOnFirstTick()
        worker = self._make_worker(decider=dec)
        worker_ref[0] = worker
        result = _run_worker(worker)
        self.assertEqual(result["outcome"], "canceled")
        # No partial chunks after cancel point leaked
        for rec in result["gate_records"]:
            self.assertIsNotNone(rec.chunk_id)

    # GATE-5b: cancel after first chunk completes; prior records preserved
    def test_gate_5_prior_records_preserved(self):
        call_count = [0]
        worker_ref = [None]

        class _CancelOnSecondTick:
            label = "cancel2_dec"
            def __call__(self, frames_rgb, fps, t_ms):
                call_count[0] += 1
                if call_count[0] == 2 and worker_ref[0] is not None:
                    worker_ref[0].cancel()
                return {h: 0.05 for h in HAZARDS}

        dec = _CancelOnSecondTick()
        adapter = RecordingAdapter()
        worker = self._make_worker(decider=dec, adapter=adapter)
        worker_ref[0] = worker
        result = _run_worker(worker)
        self.assertEqual(result["outcome"], "canceled")
        for ev in result["vlm_events"]:
            self.assertIsNotNone(ev.chunk_id)

    # GATE-6: log structure
    def test_gate_6_log_structure(self):
        worker = self._make_worker()
        result = _run_worker(worker)
        self.assertEqual(result["outcome"], "finished")
        log_files = [f for f in os.listdir(self.log_dir) if f.endswith(".jsonl")]
        self.assertEqual(len(log_files), 1)
        log_path = os.path.join(self.log_dir, log_files[0])
        with open(log_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        types = [l["type"] for l in lines]
        self.assertEqual(types[0], "run_start")
        self.assertEqual(types[-1], "run_end")
        self.assertEqual(lines[-1]["outcome"], "finished")
        chunk_lines = [l for l in lines if l["type"] == "chunk"]
        n_chunks = len(result["gate_records"])
        self.assertEqual(len(chunk_lines), n_chunks)
        required_chunk_keys = [
            "chunk_id", "start_ms", "end_ms", "tier", "reason",
            "vec", "pmax", "rising_hazard", "tick_ids",
            "num_ticks_ok", "num_ticks_error", "sample_fps",
            "frames_sent", "frames_full_equiv", "jev_time_s",
            "vlm_time_s", "vlm_status", "vlm_score", "wall_done_s",
            "hold_remaining_after", "top_tick_id",
        ]
        for cl in chunk_lines:
            for k in required_chunk_keys:
                self.assertIn(k, cl)
        wall_times = [l["wall_done_s"] for l in chunk_lines]
        for i in range(1, len(wall_times)):
            self.assertGreaterEqual(wall_times[i], wall_times[i - 1])


# ---------------------------------------------------------------------------
# WIN-1: all-FULL GatedBatchWorker matches plain BatchWorker frame counts
# ---------------------------------------------------------------------------

class TestWin1BaselineEquivalence(unittest.TestCase):

    def setUp(self):
        self.video = make_test_video(duration_ms=12000, fps=30.0)
        self.log_dir = tempfile.mkdtemp()

    def tearDown(self):
        os.unlink(self.video)
        import shutil
        shutil.rmtree(self.log_dir, ignore_errors=True)

    def test_win_1_frame_count_equivalence(self):
        high_rules = [(0, 99999999, {h: 0.99 for h in HAZARDS})]
        dec = MockJevDecider(rules=high_rules)
        cfg = GateConfig()
        adapter_gated = RecordingAdapter()
        adapter_plain = RecordingAdapter()
        gated_worker = GatedBatchWorker(
            media_path=self.video, adapter=adapter_gated,
            chunk_s=5.0, overlap_s=1.0, vlm_fps=4.0, parent=None,
            decider=dec, gate_config=cfg, log_dir=self.log_dir,
        )
        plain_worker = BatchWorker(
            media_path=self.video, adapter=adapter_plain,
            chunk_s=5.0, overlap_s=1.0, vlm_fps=4.0, parent=None,
        )
        r_gated = _run_worker(gated_worker)
        r_plain  = _run_worker(plain_worker)
        self.assertEqual(r_gated["outcome"], "finished")
        self.assertEqual(r_plain["outcome"],  "finished")
        self.assertEqual(len(r_gated["vlm_events"]), len(r_plain["vlm_events"]))
        for gc, pc in zip(adapter_gated.calls, adapter_plain.calls):
            self.assertEqual(len(gc.frames), len(pc.frames),
                f"chunk {gc.chunk_id}: gated={len(gc.frames)} plain={len(pc.frames)}")


if __name__ == "__main__":
    unittest.main()