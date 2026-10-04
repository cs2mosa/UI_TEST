from __future__ import annotations
import math


def compute_chunk_windows(
    duration_ms: int,
    chunk_s: float = 5.0,
    overlap_s: float = 1.0,
    include_partial: bool = True
) -> list[tuple[int, int]]:
    """
    Computes deterministic [start_ms, end_ms] windows for a given duration.
    
    stride_s = chunk_s - overlap_s
    Chunk k covers [k * stride_s, k * stride_s + chunk_s] (in ms).
    """
    if duration_ms <= 0:
        return []
    
    if chunk_s <= 0:
        raise ValueError(f"chunk_s must be > 0, got {chunk_s}")
    if overlap_s < 0:
        raise ValueError(f"overlap_s must be >= 0, got {overlap_s}")
    if overlap_s >= chunk_s:
        raise ValueError(f"overlap_s ({overlap_s}) must be strictly less than chunk_s ({chunk_s})")
        
    chunk_ms = int(round(chunk_s * 1000.0))
    stride_ms = int(round((chunk_s - overlap_s) * 1000.0))
    
    # If video is shorter than one chunk, return single window covering the video
    if duration_ms <= chunk_ms:
        return [(0, duration_ms)]
        
    windows: list[tuple[int, int]] = []
    current_start = 0
    
    while current_start < duration_ms:
        current_end = current_start + chunk_ms
        if current_end >= duration_ms:
            # Overhangs or touches the end
            if current_start < duration_ms:
                if include_partial or current_end == duration_ms:
                    windows.append((current_start, min(current_end, duration_ms)))
            break
        else:
            windows.append((current_start, current_end))
            current_start += stride_ms
            
    return windows
