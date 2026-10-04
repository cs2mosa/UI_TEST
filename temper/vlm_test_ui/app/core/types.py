from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np


class InvalidVLMReply(Exception):
    """Raised when the VLM output cannot be parsed into expected format."""
    def __init__(self, raw: str):
        super().__init__(f"Invalid VLM response format: {raw}")
        self.raw = raw


@dataclass(frozen=True)
class VideoChunk:
    """Represents a discrete temporal slice of video frames sent to the VLM."""
    chunk_id: int
    source: str
    start_ms: int
    end_ms: int
    fps: float                     # effective sampling fps of `frames`
    frames: list[np.ndarray]       # RGB uint8, HxWx3
    timestamps_ms: list[int]


@dataclass
class VLMResult:
    """Standardized response from a VLM model."""
    caption: str
    score: float                   # 0.0 - 1.0, severity rating


# Type alias for any user-supplied VLM callable
VLMCallable = Callable[[VideoChunk], VLMResult | tuple[str, float] | dict[str, Any]]


@dataclass
class VLMEvent:
    """Processed chunk event stored in SessionModel and rendered in UI."""
    id: int
    chunk_id: int
    start_ms: int
    end_ms: int
    latency_s: float | None = None
    latency_per_frame_ms: float | None = None  # latency_s * 1000 / frames, set on success
    caption: str | None = None
    score: float | None = None            # None on error
    status: str = "ok"                    # "ok" | "error" | "timeout" | "invalid"
    error: str | None = None
    session_reset: bool = False           # True if stateful session was reset due to timeout/failure
    thumbnails: list[bytes] = field(default_factory=list)  # JPEGs: first / middle / last
    
    # Ground truth (None when GT mode is off)
    expected_score: float | None = None
    gt_description: str | None = None
    match: str | None = None              # "MATCH" | "MISS" | "FP" | "OFF"
    
    # Tester review
    verdict: str | None = None            # "correct" | "wrong"
    failure_type: str | None = None       # "FP" | "MISS" | "SCORE" | "HALLUCINATION" | "OTHER"
    note: str = ""
