from __future__ import annotations
import os
import tempfile
from typing import Any

import cv2
import numpy as np

from app.core.types import VideoChunk, VLMEvent, VLMResult


def make_test_video(
    duration_ms: int = 12000,
    fps: float = 30.0,
    width: int = 64,
    height: int = 64,
    suffix: str = ".avi",
) -> str:
    """
    Create a synthetic video where every planned timestamp decodes successfully.

    Returns the path to a temporary file; caller is responsible for deletion.
    Unlike `examples/synthetic_test.mp4`, this video is built so that
    `read_frames_at` never hits a failed read within `duration_ms`.
    """
    n_frames = max(1, int(round(duration_ms / 1000.0 * fps)))
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))
    for i in range(n_frames):
        color = (i * 7 % 255, i * 13 % 255, i * 17 % 255)
        frame = np.full((height, width, 3), color, dtype=np.uint8).reshape(height, width, 3)
        frame[:, :] = color
        out.write(frame)
    out.release()
    return tmp.name


class RecordingAdapter:
    """
    Drop-in replacement for `VLMAdapter` in tests.

    Records every `process_chunk` call in `self.calls` and returns a
    deterministic `VLMEvent` so tests do not need a real VLM.
    """

    def __init__(self, latency_s: float = 0.0, score: float = 0.1, caption: str = "test"):
        self.latency_s = latency_s
        self.score = score
        self.caption = caption
        self.calls: list[VideoChunk] = []
        # Simulate a stateless VLM (no start_session / end_session)
        self.vlm: Any = object()

    def process_chunk(self, chunk: VideoChunk, event_id: int = 1) -> VLMEvent:
        self.calls.append(chunk)
        return VLMEvent(
            id=event_id,
            chunk_id=chunk.chunk_id,
            start_ms=chunk.start_ms,
            end_ms=chunk.end_ms,
            num_frames=len(chunk.frames),
            latency_s=self.latency_s,
            latency_per_frame_ms=(self.latency_s * 1000 / len(chunk.frames)) if chunk.frames else None,
            caption=self.caption,
            score=self.score,
            status="ok",
        )