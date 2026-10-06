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
    num_frames: int = 0
    latency_per_frame_ms: float | None = None
    caption: str | None = None
    score: float | None = None            # None on error
    status: str = "ok"                    # "ok" | "error" | "invalid"
    error: str | None = None
    thumbnails: list[bytes] = field(default_factory=list)  # JPEGs: first / middle / last
    
    # Ground truth (None when GT mode is off)
    expected_score: float | None = None
    gt_description: str | None = None
    match: str | None = None              # "MATCH" | "MISS" | "FP" | "OFF"
    
    # Tester review
    verdict: str | None = None            # "correct" | "wrong"
    failure_type: str | None = None       # "FP" | "MISS" | "SCORE" | "HALLUCINATION" | "OTHER"
    note: str = ""


# ---------------------------------------------------------------------------
# JEV gate data types (Phase 1 additions)
# ---------------------------------------------------------------------------

@dataclass
class TickRecord:
    """One JEV evaluation result."""
    tick_id: int
    t_ms: int
    window_start_ms: int
    num_frames: int
    status: str                              # "ok" | "error"
    probs: dict[str, float] | None = None
    error: str | None = None
    jev_time_s: float = 0.0
    thumbs: list[bytes] = field(default_factory=list)


@dataclass
class GateRecord:
    """Gate routing outcome for one chunk (one per chunk, emitted before VLMEvent)."""
    chunk_id: int
    start_ms: int
    end_ms: int
    tier: str                                # "FULL" | "REDUCED"
    reason: str                              # one of the six reasons
    vec: dict[str, float] | None
    pmax: float | None
    rising_hazard: str | None
    tick_ids: list[int]
    ticks: list[TickRecord]
    top_tick_id: int | None
    num_ticks_ok: int
    num_ticks_error: int
    sample_fps: float
    frames_sent: int
    frames_full_equiv: int
    jev_time_s: float
    vlm_time_s: float | None
    vlm_status: str | None
    vlm_score: float | None
    wall_done_s: float
    hold_remaining_after: int
