"""
Phase 4 tests for InferenceWorker (Task 4.5).

INF-1  accept_chunk + seal -> result_ready fires for every chunk
INF-2  gate_rec is forwarded unchanged to result_ready
INF-3  progress fires (done, total) with correct counts
INF-4  finished fires after seal()
INF-5  cancel() stops the worker; cancelled signal fires
INF-6  VLM error in one chunk -> event.status == 'error', worker continues

All tests use MockVLM (no GPU / no model required).
"""
from __future__ import annotations

import sys
import time
import unittest
import numpy as np

from PySide6.QtCore import QCoreApplication

_app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: int, num_frames: int = 2):
    from app.core.types import VideoChunk
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(num_frames)]
    return VideoChunk(
        chunk_id=chunk_id,
        source="test.mp4",
        start_ms=chunk_id * 5000,
        end_ms=(chunk_id + 1) * 5000,
        fps=1.0,
        frames=frames,
        timestamps_ms=[chunk_id * 5000 + i * 1000 for i in range(num_frames)],
    )


def _make_gate_rec(chunk_id: int):
    from app.core.jev_gate import GateRecord
    return GateRecord(
        chunk_id=chunk_id,
        start_ms=chunk_id * 5000,
        end_ms=(chunk_id + 1) * 5000,
        tier="full",
        reason="mock",
        vec={},
        pmax=0.0,
        rising_hazard=False,
        tick_ids=[],
        ticks=[],
        top_tick_id=None,
        num_ticks_ok=0,
        num_ticks_error=0,
        sample_fps=1.0,
        frames_sent=2,
        frames_full_equiv=2,
        jev_time_s=0.0,
        vlm_time_s=0.0,
        vlm_status="ok",
        vlm_score=0.0,
        wall_done_s=0.0,
        hold_remaining_after=0,
    )


class MockVLM:
    def __init__(self, score: float = 0.2, caption: str = "mock scene"):
        from app.core.types import VLMResult
        self._result = VLMResult(caption=caption, score=score)

    def __call__(self, chunk):
        return self._result


class ErrorVLM:
    def __call__(self, chunk):
        raise RuntimeError("synthetic inference failure")


def _drain(worker, timeout_ms: int = 5000) -> None:
    """
    Wait for the worker QThread to finish, then flush any queued signals.
    Using worker.wait() is race-free: it blocks until QThread::run() returns,
    then processEvents() delivers any signals that were queued during the run.
    """
    finished = worker.wait(timeout_ms)
    if not finished:
        worker.terminate()
        worker.wait(1000)
    # Two processEvents passes ensure cross-thread queued signals are delivered
    _app.processEvents()
    _app.processEvents()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestInferenceWorkerCPU(unittest.TestCase):

    def _make_worker(self, vlm=None, total_chunks: int = 0):
        from app.workers.inference_worker import InferenceWorker
        return InferenceWorker(vlm or MockVLM(), total_chunks=total_chunks)

    # INF-1 -----------------------------------------------------------------
    def test_inf_1_result_ready_fires_for_every_chunk(self):
        """result_ready is emitted once per chunk."""
        worker = self._make_worker(total_chunks=3)
        fired = []
        worker.result_ready.connect(lambda cid, gr, ev: fired.append(cid))

        worker.start()
        for i in range(3):
            worker.accept_chunk(_make_chunk(i), _make_gate_rec(i))
        worker.seal()

        _drain(worker)
        self.assertEqual(sorted(fired), [0, 1, 2])

    # INF-2 -----------------------------------------------------------------
    def test_inf_2_gate_rec_forwarded_unchanged(self):
        """The GateRecord emitted is the same object that was passed in."""
        worker = self._make_worker(total_chunks=1)
        received = []
        worker.result_ready.connect(lambda cid, gr, ev: received.append(gr))

        worker.start()
        gr = _make_gate_rec(7)
        worker.accept_chunk(_make_chunk(7), gr)
        worker.seal()

        _drain(worker)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].chunk_id, 7)
        self.assertIs(received[0], gr)

    # INF-3 -----------------------------------------------------------------
    def test_inf_3_progress_fires_with_correct_counts(self):
        """progress(done, total) increments done; total matches constructor arg."""
        N = 4
        worker = self._make_worker(total_chunks=N)
        log = []
        worker.progress.connect(lambda d, t: log.append((d, t)))

        worker.start()
        for i in range(N):
            worker.accept_chunk(_make_chunk(i), _make_gate_rec(i))
        worker.seal()

        _drain(worker)
        self.assertEqual(len(log), N)
        for idx, (done, total) in enumerate(log, 1):
            self.assertEqual(done, idx)
            self.assertEqual(total, N)

    # INF-4 -----------------------------------------------------------------
    def test_inf_4_finished_fires_after_seal(self):
        """finished signal fires exactly once after all chunks processed."""
        worker = self._make_worker(total_chunks=2)
        fin = []
        worker.run_finished.connect(lambda: fin.append(1))

        worker.start()
        for i in range(2):
            worker.accept_chunk(_make_chunk(i), _make_gate_rec(i))
        worker.seal()

        _drain(worker)
        self.assertEqual(fin, [1])

    # INF-5 -----------------------------------------------------------------
    def test_inf_5_cancel_stops_worker(self):
        """cancel() stops the worker; cancelled fires; at most 1 result emitted."""
        worker = self._make_worker(total_chunks=100)
        fired = []
        cancelled_flag = []
        worker.result_ready.connect(lambda cid, gr, ev: fired.append(cid))
        worker.cancelled.connect(lambda: cancelled_flag.append(1))

        worker.start()
        worker.accept_chunk(_make_chunk(0), _make_gate_rec(0))
        worker.cancel()
        for i in range(1, 10):
            worker.accept_chunk(_make_chunk(i), _make_gate_rec(i))

        _drain(worker, timeout_ms=4000)
        self.assertEqual(cancelled_flag, [1])
        self.assertLessEqual(len(fired), 1)

    # INF-6 -----------------------------------------------------------------
    def test_inf_6_vlm_error_in_chunk_event_status_error(self):
        """A VLM exception produces event.status=='error'; worker continues."""
        worker = self._make_worker(vlm=ErrorVLM(), total_chunks=2)
        events = []
        fin = []
        worker.result_ready.connect(lambda cid, gr, ev: events.append(ev))
        worker.run_finished.connect(lambda: fin.append(1))

        worker.start()
        for i in range(2):
            worker.accept_chunk(_make_chunk(i), _make_gate_rec(i))
        worker.seal()

        _drain(worker)
        self.assertEqual(len(events), 2)
        for ev in events:
            self.assertEqual(ev.status, "error")
        self.assertEqual(fin, [1])


if __name__ == "__main__":
    unittest.main()
