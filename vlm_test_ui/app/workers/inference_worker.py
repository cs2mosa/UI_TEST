"""
InferenceWorker — Phase 4, Task 4.2
=====================================

A QThread worker that receives VideoChunk batches from GatedBatchWorker
(via direct call or signal connection) and drives VLM inference through the
HermesStreamingRunner (or any VLMCallable mock for CPU tests).

Signal contract (SS7.2):
  result_ready(chunk_id: int, gate_rec: GateRecord, event: VLMEvent)
  progress(done: int, total: int)
  run_finished()
  error_occurred(msg: str)
  cancelled()

Usage:
    runner = HermesStreamingRunner(model_path)
    worker = InferenceWorker(runner, total_chunks=n)
    worker.result_ready.connect(ui_slot)
    worker.start()
    # Feed chunks as they arrive from GatedBatchWorker:
    gate_worker.frame_ready.connect(worker.accept_chunk)
    gate_worker.finished.connect(worker.seal)          # signals no more chunks
"""

from __future__ import annotations

import queue
import threading
from time import perf_counter
from typing import Optional

from PySide6.QtCore import QThread, Signal

from app.core.types import VideoChunk, VLMCallable, VLMEvent
from app.core.vlm_adapter import VLMAdapter
from app.core.jev_gate import GateRecord   # type: ignore[attr-defined]


# Sentinel pushed onto the queue to signal "no more chunks"
_DONE = object()


class InferenceWorker(QThread):
    """
    QThread that runs VLM inference on chunks fed by GatedBatchWorker.

    Thread safety
    -------------
    ``accept_chunk`` and ``seal`` are safe to call from *any* thread
    (including the main thread / another QThread) because they push onto a
    thread-safe queue.  The worker thread drains that queue and emits Qt
    signals back to the main thread in the normal way.
    """

    # Emitted for each processed chunk (chunk_id mirrors GateRecord.chunk_id)
    result_ready = Signal(int, object, object)   # chunk_id, GateRecord, VLMEvent
    progress     = Signal(int, int)              # done, total
    run_finished = Signal()   # fires when all chunks done (not QThread.finished)
    error_occurred = Signal(str)
    cancelled    = Signal()

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        vlm: VLMCallable,
        total_chunks: int = 0,
        *,
        hermes_session: bool = False,
        parent=None,
    ):
        """
        Parameters
        ----------
        vlm:
            Any VLMCallable — ``HermesStreamingRunner`` in production,
            ``MockVLM`` / lambda in tests.
        total_chunks:
            Expected number of chunks (used for progress reporting only).
            Pass 0 if unknown; progress will still fire but ``total`` will
            be reported as the running count.
        hermes_session:
            If True, calls ``vlm.start_session()`` / ``vlm.end_session()``
            around the entire batch (streaming mode for HERMES).
        """
        super().__init__(parent)
        self._adapter       = VLMAdapter(vlm)
        self._vlm           = vlm
        self._total         = total_chunks
        self._hermes        = hermes_session
        self._queue: queue.Queue = queue.Queue()
        self._is_canceled   = False
        self._done_event    = threading.Event()

    # ------------------------------------------------------------------ #
    # Public thread-safe API (called from outside this thread)            #
    # ------------------------------------------------------------------ #

    def accept_chunk(self, chunk: VideoChunk, gate_rec: GateRecord) -> None:
        """Enqueue a chunk for inference. Safe to call from any thread."""
        if not self._is_canceled:
            self._queue.put((chunk, gate_rec))

    def seal(self) -> None:
        """
        Signal that no more chunks are coming.
        Called when GatedBatchWorker emits ``finished`` or ``cancelled``.
        """
        self._queue.put(_DONE)

    def cancel(self) -> None:
        """Request graceful cancellation."""
        self._is_canceled = True
        self._queue.put(_DONE)   # unblock the worker if it is waiting

    # ------------------------------------------------------------------ #
    # QThread entry point                                                  #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        try:
            self._run_inference_loop()
        except Exception as exc:
            self.error_occurred.emit(f"{type(exc).__name__}: {exc}")

    def _run_inference_loop(self) -> None:
        if self._hermes:
            try:
                self._vlm.start_session()
            except AttributeError:
                pass   # mock without session lifecycle

        done = 0
        try:
            while True:
                if self._is_canceled:
                    break

                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is _DONE:
                    break

                chunk, gate_rec = item

                if self._is_canceled:
                    break

                # Run inference through the adapter (handles InvalidVLMReply etc.)
                event: VLMEvent = self._adapter.process_chunk(
                    chunk, event_id=gate_rec.chunk_id + 1
                )

                done += 1
                total = self._total if self._total > 0 else done

                self.result_ready.emit(gate_rec.chunk_id, gate_rec, event)
                self.progress.emit(done, total)

        finally:
            if self._hermes:
                try:
                    self._vlm.end_session()
                except AttributeError:
                    pass

        if self._is_canceled:
            self.cancelled.emit()
        else:
            self.run_finished.emit()
