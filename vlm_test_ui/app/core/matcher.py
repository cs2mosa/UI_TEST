from __future__ import annotations
from typing import Optional
from app.core.gt_loader import GTRange
from app.core.types import VLMEvent
from app.config import DEFAULT_CONFIG


def compute_chunk_expected_gt(
    chunk_start_ms: int,
    chunk_end_ms: int,
    gt_ranges: list[GTRange]
) -> tuple[float, str]:
    """
    Finds overlapping GT ranges for a chunk window.
    expected = max(GT score over GT ranges overlapping the chunk span), or 0 if none overlap.
    """
    overlapping = []
    for gt in gt_ranges:
        # Check temporal overlap
        overlap_start = max(chunk_start_ms, gt.start_ms)
        overlap_end = min(chunk_end_ms, gt.end_ms)
        if overlap_start < overlap_end or (gt.start_ms == gt.end_ms == 0):
            overlapping.append(gt)

    if not overlapping:
        return 0.0, "Safe (Routine)"

    # Pick range with highest severity score
    overlapping.sort(key=lambda g: g.score, reverse=True)
    best = overlapping[0]
    return best.score, best.description


def classify_match(
    vlm_score: Optional[float],
    expected_score: float,
    tol: float = DEFAULT_CONFIG.tol,
    high_thr: float = DEFAULT_CONFIG.high_thr,
) -> str:
    """
    Classifies a chunk based on Design Doc §8:
    - MATCH: abs(score - expected) <= tol
    - MISS: expected >= high_thr and score < expected - tol
    - FALSE POSITIVE: expected < high_thr and score > expected + tol
    - SCORE OFF: Other mismatches
    """
    if vlm_score is None:
        return "ERROR"

    diff = abs(vlm_score - expected_score)
    if diff <= tol:
        return "MATCH"
    elif expected_score >= high_thr and vlm_score < (expected_score - tol):
        return "MISS"
    elif expected_score < high_thr and vlm_score > (expected_score + tol):
        return "FP"
    else:
        return "OFF"


def match_event_with_gt(
    event: VLMEvent,
    gt_ranges: list[GTRange],
    tol: float = DEFAULT_CONFIG.tol,
    high_thr: float = DEFAULT_CONFIG.high_thr,
) -> VLMEvent:
    """Populates expected_score, gt_description, and match verdict on a VLMEvent."""
    expected_score, description = compute_chunk_expected_gt(event.start_ms, event.end_ms, gt_ranges)
    event.expected_score = expected_score
    event.gt_description = description
    event.match = classify_match(event.score, expected_score, tol=tol, high_thr=high_thr)
    return event
