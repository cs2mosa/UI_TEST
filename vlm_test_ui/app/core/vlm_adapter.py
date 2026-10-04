from __future__ import annotations
import time
import io
from typing import Any
import numpy as np
from PIL import Image

from app.core.types import VideoChunk, VLMResult, VLMEvent, InvalidVLMReply, VLMCallable


def frame_to_jpeg_bytes(frame_rgb: np.ndarray, max_dim: int = 160) -> bytes:
    """Resize RGB numpy array maintaining aspect ratio and compress to JPEG bytes."""
    try:
        h, w = frame_rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = Image.fromarray(frame_rgb).resize((new_w, new_h), Image.Resampling.BILINEAR)
        else:
            img = Image.fromarray(frame_rgb)
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return buf.getvalue()
    except Exception:
        return b""


def create_chunk_thumbnails(frames: list[np.ndarray]) -> list[bytes]:
    """Generates first, middle, and last thumbnail JPEGs for a chunk."""
    if not frames:
        return []
    indices = [0]
    if len(frames) > 2:
        indices.append(len(frames) // 2)
    if len(frames) > 1:
        indices.append(len(frames) - 1)
    
    # Preserve order and deduplicate indices
    seen = set()
    uniq_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            uniq_indices.append(idx)
            
    return [frame_to_jpeg_bytes(frames[i]) for i in uniq_indices]


class VLMAdapter:
    """
    Wraps any user-provided VLMCallable.
    Validates output, handles timeouts, enforces error boundaries, and normalizes output into VLMEvent.
    """
    def __init__(self, vlm: VLMCallable, *args, **kwargs):
        self.vlm = vlm

    def _normalize_output(self, raw_output: Any) -> tuple[str, float]:
        """Validates and normalizes raw output to (caption, score)."""
        if isinstance(raw_output, VLMResult):
            caption, score = raw_output.caption, raw_output.score
        elif isinstance(raw_output, (tuple, list)) and len(raw_output) == 2:
            caption, score = raw_output[0], raw_output[1]
        elif isinstance(raw_output, dict) and "caption" in raw_output and "score" in raw_output:
            caption, score = raw_output["caption"], raw_output["score"]
        else:
            raise InvalidVLMReply(f"Expected VLMResult, tuple, or dict, got: {type(raw_output)}")

        try:
            score = float(score)
        except (ValueError, TypeError):
            raise InvalidVLMReply(f"Score could not be parsed as float: {score}")

        caption = str(caption).strip()
        return caption, score

    def process_chunk(self, chunk: VideoChunk, event_id: int) -> VLMEvent:
        """Processes a single chunk, measures latency, and calculates per-frame latency."""
        start_time = time.perf_counter()
        thumbnails = create_chunk_thumbnails(chunk.frames)
        num_frames = len(chunk.frames)
        
        event = VLMEvent(
            id=event_id,
            chunk_id=chunk.chunk_id,
            start_ms=chunk.start_ms,
            end_ms=chunk.end_ms,
            num_frames=num_frames,
            thumbnails=thumbnails,
        )

        try:
            raw_output = self.vlm(chunk)
            event.latency_s = time.perf_counter() - start_time
            if num_frames > 0:
                event.latency_per_frame_ms = (event.latency_s / num_frames) * 1000.0
            
            caption, score = self._normalize_output(raw_output)
            event.caption = caption
            event.score = score
            
            # Check range [0.0, 1.0]. If out of range: flagged, not clamped (Design doc §6)
            if score < 0.0 or score > 1.0:
                event.status = "invalid"
                event.error = f"Score {score} is out of expected range [0.0, 1.0]"
            else:
                event.status = "ok"

        except InvalidVLMReply as e:
            event.latency_s = time.perf_counter() - start_time
            if num_frames > 0:
                event.latency_per_frame_ms = (event.latency_s / num_frames) * 1000.0
            event.status = "invalid"
            event.error = str(e)
            event.caption = e.raw
        except Exception as e:
            event.latency_s = time.perf_counter() - start_time
            if num_frames > 0:
                event.latency_per_frame_ms = (event.latency_s / num_frames) * 1000.0
            event.status = "error"
            event.error = f"{type(e).__name__}: {str(e)}"

        return event

    def shutdown(self):
        """No-op kept for lifecycle compatibility."""
        pass
