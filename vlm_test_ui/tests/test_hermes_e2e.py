"""
End-to-end integration tests for the HERMES → VLMAdapter → BatchWorker pipeline.

Requires:
  - CUDA GPU with ≥6 GB VRAM
  - Qwen3-VL-4B-Instruct downloaded
  - PySide6 installed

Run with:
  python -m pytest tests/test_hermes_e2e.py -v --tb=short -x
"""
import os
import unittest
import time
import numpy as np
import torch
import pytest

# Point HF to the local cache so no network calls are made
os.environ.setdefault("HF_HOME", r"F:\URCA_PROJECTS\.hf_cache")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from PySide6.QtWidgets import QApplication

gpu_available = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required"
)

# Single QApplication instance
_app = QApplication.instance() or QApplication([])

MODEL_PATH = (
    r"F:\URCA_PROJECTS\.hf_cache\hub"
    r"\models--Qwen--Qwen3-VL-4B-Instruct"
    r"\snapshots\ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
KV_SIZE = 2000

# The 6 GB GPU holds only ONE 4-bit model instance. All tests share this
# runner so the model is loaded exactly once per pytest run and tests use
# it strictly one at a time (each within its own start/end_session scope).
_shared_runner = None


def _get_shared_runner():
    global _shared_runner
    if _shared_runner is None:
        from vlm.hermes_runner import HermesStreamingRunner
        _shared_runner = HermesStreamingRunner(MODEL_PATH, kv_size=KV_SIZE, max_new_tokens=60)
    return _shared_runner


def _make_dummy_chunk(chunk_id=0, num_frames=4, h=64, w=64):
    from app.core.types import VideoChunk
    frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(num_frames)]
    return VideoChunk(
        chunk_id=chunk_id,
        source="dummy.mp4",
        start_ms=chunk_id * 5000,
        end_ms=(chunk_id + 1) * 5000,
        fps=4.0,
        frames=frames,
        timestamps_ms=list(range(chunk_id * 5000, (chunk_id + 1) * 5000, 1000))[:num_frames],
    )


class TestHermesRunnerCallable(unittest.TestCase):

    def tearDown(self):
        # Belt-and-braces: release any session state the test left behind so
        # the next test starts from a clean cache on the shared model.
        if _shared_runner is not None:
            try:
                _shared_runner.end_session()
            except Exception:
                pass
            torch.cuda.empty_cache()

    @gpu_available
    def test_runner_callable_returns_vlm_result(self):
        """HermesStreamingRunner produces a valid VLMResult from one chunk."""
        from app.core.types import VLMResult
        runner = _get_shared_runner()
        chunk = _make_dummy_chunk(0)

        runner.start_session()
        result = runner(chunk)
        runner.end_session()

        self.assertIsInstance(result, VLMResult)
        self.assertIsInstance(result.caption, str)
        self.assertIsInstance(result.score, float)

    @gpu_available
    def test_multi_chunk_session_stable_memory(self):
        """3 sequential chunks: all succeed, GPU memory stays under 6 GB."""
        runner = _get_shared_runner()
        runner.start_session()

        torch.cuda.reset_peak_memory_stats()
        for i in range(3):
            chunk = _make_dummy_chunk(chunk_id=i)
            result = runner(chunk)
            self.assertIsNotNone(result)

        runner.end_session()
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        self.assertLess(peak_gb, 6.0, f"Peak VRAM {peak_gb:.2f} GB exceeds 6 GB budget")

    @gpu_available
    def test_adapter_integration(self):
        """VLMAdapter wrapping HermesStreamingRunner returns a proper VLMEvent."""
        from app.core.vlm_adapter import VLMAdapter
        runner = _get_shared_runner()
        adapter = VLMAdapter(runner)

        runner.start_session()
        chunk = _make_dummy_chunk(0)
        event = adapter.process_chunk(chunk, event_id=1)
        runner.end_session()

        self.assertIn(event.status, ("ok", "invalid"))  # invalid = out-of-range score but parseable
        self.assertIsNotNone(event.latency_s)
        self.assertGreater(event.latency_s, 0)
        adapter.shutdown()

    @gpu_available
    def test_batch_worker_with_synthetic_video(self):
        """
        Full BatchWorker run on synthetic_test.mp4 with HermesStreamingRunner.
        All events should have status 'ok', be in temporal order.
        """
        import os
        from app.core.vlm_adapter import VLMAdapter
        from app.workers.batch_worker import BatchWorker

        video_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples", "synthetic_test.mp4"
        )
        if not os.path.exists(video_path):
            self.skipTest(f"synthetic_test.mp4 not found at {video_path}")

        runner = _get_shared_runner()
        adapter = VLMAdapter(runner)

        events = []
        worker = BatchWorker(
            media_path=video_path,
            adapter=adapter,
            chunk_s=5.0,
            overlap_s=1.0,
            vlm_fps=0.5,
        )
        worker.event_ready.connect(lambda ev: events.append(ev))

        worker.start()
        finished = worker.wait(300_000)  # 5-minute timeout
        _app.processEvents()

        self.assertTrue(finished, "BatchWorker did not finish within 5 minutes")
        self.assertGreater(len(events), 0)

        # Events should be in temporal order
        starts = [e.start_ms for e in events]
        self.assertEqual(starts, sorted(starts))

        # All ok or invalid (not error/timeout)
        for ev in events:
            self.assertIn(ev.status, ("ok", "invalid"), f"Unexpected status: {ev.status}")

        adapter.shutdown()


if __name__ == "__main__":
    unittest.main()
