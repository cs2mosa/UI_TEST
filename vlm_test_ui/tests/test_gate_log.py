from __future__ import annotations
import json
import os
import tempfile
import unittest
from datetime import datetime

from app.core.gate_log import GateLog, make_log_path
from app.core.jev_gate import gate_record_to_dict, GateConfig, HAZARDS
from app.core.types import GateRecord, TickRecord


def _make_gate_record(chunk_id=0) -> GateRecord:
    tick = TickRecord(
        tick_id=0, t_ms=3000, window_start_ms=0,
        num_frames=4, status="ok",
        probs={h: 0.05 for h in HAZARDS},
        error=None, jev_time_s=0.01,
    )
    return GateRecord(
        chunk_id=chunk_id,
        start_ms=0, end_ms=5000,
        tier="FULL", reason="jev_incomplete",
        vec=None, pmax=None, rising_hazard=None,
        tick_ids=[0], ticks=[tick], top_tick_id=0,
        num_ticks_ok=1, num_ticks_error=0,
        sample_fps=4.0, frames_sent=20, frames_full_equiv=20,
        jev_time_s=0.01, vlm_time_s=None, vlm_status=None,
        vlm_score=None, wall_done_s=1.0, hold_remaining_after=0,
    )


# ---------------------------------------------------------------------------
# LOG-1: make_log_path sanitisation
# ---------------------------------------------------------------------------

class TestMakeLogPath(unittest.TestCase):

    def test_log_1_normal_name(self):
        now = datetime(2026, 10, 5, 14, 30, 0)
        path = make_log_path("/tmp/gate_logs", "C:/videos/my_video.mp4", now)
        basename = os.path.basename(path)
        self.assertTrue(basename.startswith("gate_20261005_143000_"))
        self.assertIn("my_video", basename)
        self.assertTrue(basename.endswith(".jsonl"))

    def test_log_1_special_chars_sanitised(self):
        now = datetime(2026, 10, 5, 14, 30, 0)
        path = make_log_path("/logs", "C:\\My Video File (2026).mp4", now)
        basename = os.path.basename(path)
        # Special chars should be replaced with underscores
        self.assertNotIn(" ", basename)
        self.assertNotIn("(", basename)
        self.assertNotIn(")", basename)
        self.assertTrue(basename.endswith(".jsonl"))

    def test_log_1_empty_stem_uses_media(self):
        now = datetime(2026, 10, 5, 14, 30, 0)
        # Edge: media_path whose basename after splitext gives empty stem
        # os.path.splitext("") -> ("", "") so stem="" -> "media"
        path = make_log_path("/logs", "", now)
        basename = os.path.basename(path)
        # stem is empty -> use "media" fallback
        self.assertIn("media", basename)

    def test_log_1_path_uses_log_dir(self):
        now = datetime(2026, 10, 5, 14, 30, 0)
        path = make_log_path("/my/log/dir", "video.mp4", now)
        self.assertTrue(path.startswith("/my/log/dir"))


# ---------------------------------------------------------------------------
# LOG-2: GateLog flush-per-write
# ---------------------------------------------------------------------------

class TestGateLogFlush(unittest.TestCase):

    def test_log_2_flush_per_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.jsonl")
            log = GateLog(path)
            log.write({"type": "run_start", "media": "x.mp4"})
            # File should be readable (flushed) before close
            with open(path, "r") as f:
                lines = f.read().strip().splitlines()
            self.assertEqual(len(lines), 1)
            obj = json.loads(lines[0])
            self.assertEqual(obj["type"], "run_start")
            log.close()

    def test_log_2_multiple_writes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.jsonl")
            log = GateLog(path)
            for i in range(3):
                log.write({"type": "tick", "tick_id": i})
            with open(path, "r") as f:
                lines = f.read().strip().splitlines()
            self.assertEqual(len(lines), 3)
            log.close()


# ---------------------------------------------------------------------------
# LOG-3: OSError -> failed=True, returns False, never raises
# ---------------------------------------------------------------------------

class TestGateLogOSError(unittest.TestCase):

    def test_log_3_oserror_on_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.jsonl")
            log = GateLog(path)
            # Write one good record
            ok = log.write({"type": "start"})
            self.assertTrue(ok)
            # Force a write failure by closing the internal file handle
            log._file.close()
            result = log.write({"type": "tick"})
            self.assertFalse(result)
            self.assertTrue(log.failed)
            self.assertIsNotNone(log.error)

    def test_log_3_oserror_on_open(self):
        # Try to open a file in a non-existent nested directory that can't be created
        # This should NOT raise; GateLog handles it gracefully
        path = os.path.join(tempfile.gettempdir(), "gate_log_test_dir", "sub", "gate.jsonl")
        # It SHOULD create the directory (make_log_path implies mkdir)
        log = GateLog(path)
        result = log.write({"type": "start"})
        self.assertTrue(result)  # makedirs worked
        log.close()
        # Clean up
        import shutil
        shutil.rmtree(os.path.join(tempfile.gettempdir(), "gate_log_test_dir"), ignore_errors=True)

    def test_log_3_subsequent_writes_skipped_after_failure(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.jsonl")
            log = GateLog(path)
            log._file.close()
            log.write({"type": "tick1"})  # fails -> failed=True
            # subsequent writes should also return False immediately
            result = log.write({"type": "tick2"})
            self.assertFalse(result)

    def test_log_3_close_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.jsonl")
            log = GateLog(path)
            log._file.close()
            try:
                log.close()
            except Exception as e:
                self.fail(f"close() raised {e}")


# ---------------------------------------------------------------------------
# LOG-4: gate_record_to_dict key order matches §6.4
# ---------------------------------------------------------------------------

class TestGateRecordDictKeyOrder(unittest.TestCase):

    EXPECTED_CHUNK_KEYS = [
        "chunk_id", "start_ms", "end_ms", "tier", "reason",
        "vec", "pmax", "rising_hazard", "tick_ids",
        "num_ticks_ok", "num_ticks_error", "sample_fps",
        "frames_sent", "frames_full_equiv", "jev_time_s",
        "vlm_time_s", "vlm_status", "vlm_score", "wall_done_s",
        "hold_remaining_after", "top_tick_id",
        # design_v3 §6: trailing "ewma" is the documented v3 extension of the v2 key order
        "ewma",
    ]

    def test_log_4_chunk_key_order(self):
        rec = _make_gate_record()
        d = gate_record_to_dict(rec, include_ticks=False)
        keys = list(d.keys())
        self.assertEqual(keys, self.EXPECTED_CHUNK_KEYS)

    def test_log_4_include_ticks_adds_ticks_key(self):
        rec = _make_gate_record()
        d = gate_record_to_dict(rec, include_ticks=True)
        self.assertIn("ticks", d)
        self.assertIsInstance(d["ticks"], list)

    def test_log_4_json_serialisable(self):
        rec = _make_gate_record()
        d = gate_record_to_dict(rec, include_ticks=True)
        # Should not raise
        json.dumps(d, separators=(",", ":"), allow_nan=False)

    def test_log_4_no_ticks_key_when_false(self):
        rec = _make_gate_record()
        d = gate_record_to_dict(rec, include_ticks=False)
        self.assertNotIn("ticks", d)

    def test_log_4_gate_log_write_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gate.jsonl")
            log = GateLog(path)
            rec = _make_gate_record()
            obj = {"type": "chunk", "chunk_id": rec.chunk_id, **gate_record_to_dict(rec, False)}
            log.write(obj)
            log.close()
            with open(path) as f:
                line = f.readline()
            parsed = json.loads(line)
            self.assertEqual(parsed["type"], "chunk")
            self.assertEqual(parsed["chunk_id"], 0)


if __name__ == "__main__":
    unittest.main()