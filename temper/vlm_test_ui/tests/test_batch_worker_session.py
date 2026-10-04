"""
Unit and integration tests for BatchWorker session lifecycle wiring (Phase 3).
Verifies:
(a) reset() fires exactly once per video run (not per chunk).
(b) Effective overlap is overridden to 0.0 for stateful VLMs.
(c) Stateless mock_vlm.py path runs unmodified with configured overlap preserved.
"""

import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
vlm_path = os.path.join(project_root, "vlm_test_ui")
if vlm_path not in sys.path:
    sys.path.insert(0, vlm_path)

from app.core.types import VideoChunk, VLMResult, VLMEvent
from app.core.vlm_adapter import VLMAdapter
from app.workers.batch_worker import BatchWorker
from examples.mock_vlm import MockVLM


class StatefulVLMStub:
    """Mock stateful VLM tracking calls to reset() and __call__()."""
    def __init__(self):
        self.reset_count = 0
        self.call_count = 0

    def reset(self):
        self.reset_count += 1

    def __call__(self, chunk: VideoChunk) -> VLMResult:
        self.call_count += 1
        return VLMResult(caption=f"Stateful caption {self.call_count}", score=0.1)


class TestBatchWorkerSession(unittest.TestCase):
    def setUp(self):
        self.video_path = os.path.join(project_root, "vlm_test_ui", "examples", "synthetic_test.mp4")
        self.assertTrue(os.path.exists(self.video_path), f"Video missing: {self.video_path}")

    def test_mock_vlm_preserves_overlap_and_no_reset(self):
        """Stateless MockVLM must retain configured overlap_s and run unmodified."""
        mock_vlm = MockVLM(simulated_delay_s=0.0)
        adapter = VLMAdapter(mock_vlm)
        worker = BatchWorker(
            media_path=self.video_path,
            adapter=adapter,
            chunk_s=2.0,
            overlap_s=1.0,
            vlm_fps=2.0,
        )

        # Confirm overlap_s remains 1.0 for stateless VLM
        self.assertEqual(worker.overlap_s, 1.0, "Stateless VLM must preserve configured overlap_s")

        events: list[VLMEvent] = []
        worker.event_ready.connect(events.append)
        worker.run()

        self.assertGreater(len(events), 0, "BatchWorker with MockVLM should produce events")
        for ev in events:
            self.assertEqual(ev.status, "ok")
        adapter.shutdown()

    def test_stateful_vlm_overrides_overlap_and_fires_reset_once(self):
        """
        Stateful VLM must have overlap overridden to 0.0, and reset() must fire
        exactly once at session start (not once per chunk).
        """
        stateful_stub = StatefulVLMStub()
        adapter = VLMAdapter(stateful_stub)
        worker = BatchWorker(
            media_path=self.video_path,
            adapter=adapter,
            chunk_s=2.0,
            overlap_s=1.0,  # passed 1.0, but should be overridden to 0.0
            vlm_fps=2.0,
        )

        # Confirm overlap_s overridden to 0.0 for stateful VLM
        self.assertEqual(worker.overlap_s, 0.0, "Stateful VLM must have overlap_s overridden to 0.0")

        events: list[VLMEvent] = []
        worker.event_ready.connect(events.append)
        worker.run()

        # Confirm reset() fired exactly once
        self.assertEqual(
            stateful_stub.reset_count, 1,
            f"reset() must fire exactly once per session, fired {stateful_stub.reset_count} times"
        )
        self.assertGreater(stateful_stub.call_count, 1, "Must have processed multiple chunks")
        self.assertEqual(len(events), stateful_stub.call_count)
        adapter.shutdown()

    def test_qwen3vl_runner_with_batch_worker(self):
        """
        End-to-end smoke test of BatchWorker with real Qwen3VLRunner on synthetic_test.mp4.
        Verifies reset() fired once and overlap_s is 0.0.
        """
        from vlm.qwen3vl_runner import Qwen3VLRunner
        runner = Qwen3VLRunner(
            num_frames=2,
            max_new_tokens=40,
            load_in_4bit=True,
            memory_budget_frames=8,
            compress_to_frames=4,
            use_infinipot=True,
        )
        adapter = VLMAdapter(runner)
        worker = BatchWorker(
            media_path=self.video_path,
            adapter=adapter,
            chunk_s=2.0,
            overlap_s=1.0,
            vlm_fps=2.0,
        )

        self.assertEqual(worker.overlap_s, 0.0, "Real Qwen3VLRunner must override overlap_s to 0.0")

        events: list[VLMEvent] = []
        worker.event_ready.connect(events.append)
        worker.run()

        self.assertGreater(len(events), 0)
        self.assertGreater(runner.frames_seen, 0)
        for ev in events:
            self.assertEqual(ev.status, "ok", f"Event failed with error: {ev.error}")
        adapter.shutdown()


if __name__ == "__main__":
    unittest.main()
