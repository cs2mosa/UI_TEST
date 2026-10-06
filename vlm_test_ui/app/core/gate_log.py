from __future__ import annotations
"""
gate_log.py - JSONL run log for the gated batch worker (§6.4).
"""
import json
import os
import re
from datetime import datetime


def make_log_path(log_dir: str, media_path: str, now: datetime) -> str:
    """
    Builds: <log_dir>/gate_<YYYYMMDD_HHMMSS>_<stem>.jsonl
    stem = sanitised basename (no extension); chars not in [A-Za-z0-9._-] -> '_'.
    Empty stem (e.g. ".mp4") -> "media".
    """
    basename = os.path.basename(media_path.replace("\\", "/"))
    stem = os.path.splitext(basename)[0]
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem) or "media"
    return os.path.join(log_dir, f"gate_{now:%Y%m%d_%H%M%S}_{stem}.jsonl")


class GateLog:
    """
    UTF-8 JSONL writer: one JSON object per line, flushed after every write.
    Never raises on I/O errors; sets failed=True and returns False instead.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.failed: bool = False
        self.error: str | None = None
        self._file = None
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._file = open(path, "w", encoding="utf-8")
        except OSError as exc:
            self.failed = True
            self.error = str(exc)

    def write(self, obj: dict) -> bool:
        """Serialize obj as JSON, write a line, flush. Returns True on success."""
        if self.failed:
            return False
        try:
            line = json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n"
            self._file.write(line)
            self._file.flush()
            return True
        except (OSError, ValueError) as exc:
            self.failed = True
            self.error = str(exc)
            return False

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass