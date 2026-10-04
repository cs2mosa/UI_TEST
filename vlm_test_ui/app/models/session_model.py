from __future__ import annotations
from typing import Optional
from app.core.types import VLMEvent

class SessionModel:
    """
    In-memory state and data model for a test session.
    Tracks all processed chunks, events, manual verdicts, and active playback lookups.
    """
    def __init__(self, media_path: str = ""):
        self.media_path: str = media_path
        self.duration_ms: int = 0
        self.fps: float = 0.0
        self.events: list[VLMEvent] = []
        self.active_event_id: Optional[int] = None
        
        # Ground truth status
        self.gt_enabled: bool = False
        self.gt_filename: Optional[str] = None
        self.gt_error: Optional[str] = None

    def clear(self):
        self.events.clear()
        self.active_event_id = None

    def add_event(self, event: VLMEvent):
        # Insert or replace
        for i, existing in enumerate(self.events):
            if existing.chunk_id == event.chunk_id:
                self.events[i] = event
                return
        self.events.append(event)
        self.events.sort(key=lambda e: e.start_ms)

    def get_event_by_id(self, event_id: int) -> Optional[VLMEvent]:
        for e in self.events:
            if e.id == event_id:
                return e
        return None

    def get_event_by_chunk_id(self, chunk_id: int) -> Optional[VLMEvent]:
        for e in self.events:
            if e.chunk_id == chunk_id:
                return e
        return None

    def get_active_event_at_ts(self, ts_ms: int) -> Optional[VLMEvent]:
        """
        Design Doc §9:
        Active chunk = the chunk covering the playhead whose start is latest
        (ties broken by score). Overlap regions therefore show the newest window's caption.
        """
        candidates = [
            e for e in self.events
            if e.start_ms <= ts_ms <= e.end_ms and e.status in ("ok", "invalid")
        ]
        if not candidates:
            # Fallback to any chunk covering timestamp even if error
            candidates = [e for e in self.events if e.start_ms <= ts_ms <= e.end_ms]

        if not candidates:
            return None

        # Sort: latest start_ms first, then highest score
        candidates.sort(key=lambda e: (e.start_ms, e.score if e.score is not None else -1.0), reverse=True)
        return candidates[0]

    def get_revealed_events(self, ts_ms: int) -> list[VLMEvent]:
        """Returns chunks that have ended at or before the playhead (reveal-as-played)."""
        return [e for e in self.events if e.end_ms <= ts_ms]

    def get_summary_metrics(self) -> dict:
        """Calculates aggregated session metrics."""
        total_chunks = len(self.events)
        valid_scores = [e.score for e in self.events if e.score is not None and e.status == "ok"]
        max_score = max(valid_scores) if valid_scores else 0.0
        avg_score = (sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0
        
        latencies = [e.latency_s for e in self.events if e.latency_s is not None]
        avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
        total_latency = sum(latencies)

        total_frames = sum(e.num_frames for e in self.events if e.num_frames > 0 and e.status == "ok")
        avg_latency_per_frame_ms = ((total_latency / total_frames) * 1000.0) if total_frames > 0 else 0.0
        fps = (total_frames / total_latency) if total_latency > 0 else 0.0

        high_severity_count = sum(1 for s in valid_scores if s >= 0.66)
        medium_severity_count = sum(1 for s in valid_scores if 0.3 <= s < 0.66)
        low_severity_count = sum(1 for s in valid_scores if s < 0.3)

        return {
            "total_chunks": total_chunks,
            "total_frames": total_frames,
            "max_score": max_score,
            "avg_score": avg_score,
            "avg_latency_s": avg_latency,
            "avg_latency_per_frame_ms": avg_latency_per_frame_ms,
            "fps": fps,
            "total_latency_s": total_latency,
            "high_severity_count": high_severity_count,
            "medium_severity_count": medium_severity_count,
            "low_severity_count": low_severity_count,
            "error_count": sum(1 for e in self.events if e.status in ("error", "invalid")),
        }
