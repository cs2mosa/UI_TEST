from __future__ import annotations
import time
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from app.core.types import VideoChunk, VLMEvent
from app.core.chunker import compute_chunk_windows
from app.core.vlm_adapter import VLMAdapter


class BatchWorker(QThread):
    """
    Background QThread that chunks video/image media, invokes VLM inference sequentially,
    and streams events and progress back to the main UI.
    Supports cancellation while preserving already-completed chunk events.
    """
    event_ready = Signal(VLMEvent)
    progress = Signal(int, int, float)    # completed_count, total_count, eta_seconds
    finished = Signal()
    canceled = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        media_path: str,
        adapter: VLMAdapter,
        chunk_s: float = 5.0,
        overlap_s: float = 1.0,
        vlm_fps: float = 4.0,
        parent=None,
    ):
        super().__init__(parent)
        self.media_path = media_path
        self.adapter = adapter
        self.chunk_s = chunk_s
        self.vlm_fps = vlm_fps
        self._is_canceled = False

        # If the VLM runner is stateful (maintains continuous KV cache across chunks),
        # chunk overlap is redundant and causes duplicate frame processing.
        # We override effective overlap_s to 0.0, while preserving user-configured overlap
        # for stateless runners (e.g. mock_vlm.py).
        if hasattr(self.adapter.vlm, "reset") and callable(self.adapter.vlm.reset):
            self.overlap_s = 0.0
        else:
            self.overlap_s = overlap_s

    def cancel(self):
        self._is_canceled = True

    def _sample_frames_for_window(
        self,
        cap: cv2.VideoCapture,
        source_fps: float,
        start_ms: int,
        end_ms: int,
    ) -> tuple[list[np.ndarray], list[int]]:
        """
        Samples frames at vlm_fps within the given [start_ms, end_ms] range.
        Frames are converted to RGB uint8.
        """
        duration_s = (end_ms - start_ms) / 1000.0
        target_frame_count = max(2, int(round(duration_s * self.vlm_fps)))
        
        # Calculate target sample timestamps within window
        sample_ts_list = np.linspace(start_ms, end_ms, target_frame_count, dtype=int).tolist()
        
        frames: list[np.ndarray] = []
        timestamps: list[int] = []

        for ts in sample_ts_list:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts)
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            actual_ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            frames.append(rgb)
            timestamps.append(actual_ts)

        # Fallback if sampling missed: grab at least one frame
        if not frames:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamps.append(start_ms)

        return frames, timestamps

    def run(self):
        try:
            # Reset persistent VLM session state at start of media processing
            self.adapter.reset_vlm_session()

            # Check if input is image
            test_img = cv2.imread(self.media_path)
            if test_img is not None:
                rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
                chunk = VideoChunk(
                    chunk_id=0,
                    source=self.media_path,
                    start_ms=0,
                    end_ms=0,
                    fps=1.0,
                    frames=[rgb],
                    timestamps_ms=[0],
                )
                self.progress.emit(0, 1, 0.0)
                event = self.adapter.process_chunk(chunk, event_id=1)
                self.event_ready.emit(event)
                self.progress.emit(1, 1, 0.0)
                self.finished.emit()
                return

            cap = cv2.VideoCapture(self.media_path)
            if not cap.isOpened():
                self.error_occurred.emit(f"Failed to open video: {self.media_path}")
                return

            source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration_ms = int(round((frame_count / source_fps) * 1000.0))

            windows = compute_chunk_windows(
                duration_ms=duration_ms,
                chunk_s=self.chunk_s,
                overlap_s=self.overlap_s,
                include_partial=True,
            )

            total_chunks = len(windows)
            if total_chunks == 0:
                cap.release()
                self.finished.emit()
                return

            start_perf = time.perf_counter()

            for idx, (w_start, w_end) in enumerate(windows):
                if self._is_canceled:
                    break

                frames, timestamps = self._sample_frames_for_window(
                    cap, source_fps, w_start, w_end
                )

                chunk = VideoChunk(
                    chunk_id=idx,
                    source=self.media_path,
                    start_ms=w_start,
                    end_ms=w_end,
                    fps=self.vlm_fps,
                    frames=frames,
                    timestamps_ms=timestamps,
                )

                # Process chunk through adapter (only current frames held in memory)
                event = self.adapter.process_chunk(chunk, event_id=idx + 1)
                self.event_ready.emit(event)

                completed = idx + 1
                elapsed = time.perf_counter() - start_perf
                avg_time_per_chunk = elapsed / completed
                remaining_chunks = total_chunks - completed
                eta_s = avg_time_per_chunk * remaining_chunks

                self.progress.emit(completed, total_chunks, eta_s)

            cap.release()

            if self._is_canceled:
                self.canceled.emit()
            else:
                self.finished.emit()

        except Exception as e:
            self.error_occurred.emit(str(e))
