from __future__ import annotations
import json
import csv
import datetime
import os
from typing import Any
from app.models.session_model import SessionModel
from app.config import AppConfig

class SessionExporter:
    """Exports session results to JSON and CSV formats."""

    @staticmethod
    def export_json(
        filepath: str,
        session: SessionModel,
        config: AppConfig,
        model_label: str = "quantized-qwen-vl",
        extra_meta: dict[str, Any] | None = None
    ):
        header = {
            "source": session.media_path,
            "duration_ms": session.duration_ms,
            "fps": session.fps,
            "date": datetime.datetime.now().isoformat(),
            "model_label": model_label,
            "chunk_settings": {
                "chunk_s": config.default_chunk_s,
                "overlap_s": config.default_overlap_s,
                "vlm_fps": config.default_vlm_fps,
            },
            "ground_truth_file": session.gt_filename if session.gt_enabled else None,
            "tolerances": {
                "tol": config.tol,
                "high_thr": config.high_thr,
            },
            "summary": session.get_summary_metrics(),
        }
        if extra_meta:
            header.update(extra_meta)

        events_data = []
        for e in session.events:
            ev = {
                "id": e.id,
                "chunk_id": e.chunk_id,
                "start_ms": e.start_ms,
                "end_ms": e.end_ms,
                "num_frames": e.num_frames,
                "latency_s": e.latency_s,
                "latency_per_frame_ms": e.latency_per_frame_ms,
                "caption": e.caption,
                "score": e.score,
                "status": e.status,
                "error": e.error,
                "verdict": e.verdict,
                "failure_type": e.failure_type,
                "note": e.note,
            }
            if session.gt_enabled:
                ev["expected_score"] = e.expected_score
                ev["gt_description"] = e.gt_description
                ev["match"] = e.match
            events_data.append(ev)

        payload = {
            "session": header,
            "events": events_data,
        }

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def export_csv(filepath: str, session: SessionModel):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        fieldnames = [
            "id", "chunk_id", "start_ms", "end_ms", "num_frames",
            "latency_s", "latency_per_frame_ms", "caption", "score", "status", "error",
            "verdict", "failure_type", "note"
        ]
        if session.gt_enabled:
            fieldnames.extend(["expected_score", "gt_description", "match"])

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in session.events:
                row = {
                    "id": e.id,
                    "chunk_id": e.chunk_id,
                    "start_ms": e.start_ms,
                    "end_ms": e.end_ms,
                    "num_frames": e.num_frames,
                    "latency_s": f"{e.latency_s:.3f}" if e.latency_s is not None else "",
                    "latency_per_frame_ms": f"{e.latency_per_frame_ms:.2f}" if e.latency_per_frame_ms is not None else "",
                    "caption": e.caption or "",
                    "score": f"{e.score:.3f}" if e.score is not None else "",
                    "status": e.status,
                    "error": e.error or "",
                    "verdict": e.verdict or "",
                    "failure_type": e.failure_type or "",
                    "note": e.note or "",
                }
                if session.gt_enabled:
                    row["expected_score"] = f"{e.expected_score:.3f}" if e.expected_score is not None else ""
                    row["gt_description"] = e.gt_description or ""
                    row["match"] = e.match or ""
                writer.writerow(row)
