from __future__ import annotations
import csv
import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GTRange:
    start_ms: int
    end_ms: int
    score: float
    description: str


class GTLoader:
    """Parses and validates ground truth CSV or JSON files."""

    @staticmethod
    def load(filepath: str, label_map: dict[str, float] | None = None) -> list[GTRange]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GT file not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".json":
            return GTLoader._load_json(filepath, label_map)
        else:
            return GTLoader._load_csv(filepath, label_map)

    @staticmethod
    def _parse_score(raw: str, label_map: dict[str, float] | None) -> float:
        raw_clean = str(raw).strip()
        try:
            return float(raw_clean)
        except ValueError:
            if label_map and raw_clean.lower() in label_map:
                return label_map[raw_clean.lower()]
            raise ValueError(f"Cannot parse GT score '{raw_clean}' as float or mapped label")

    @staticmethod
    def _load_csv(filepath: str, label_map: dict[str, float] | None) -> list[GTRange]:
        ranges: list[GTRange] = []
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=2):
                start_raw = row.get("start_ms", "").strip()
                end_raw = row.get("end_ms", "").strip()
                score_raw = row.get("score", "").strip()
                desc = row.get("description", "").strip()

                if not score_raw:
                    continue

                try:
                    score = GTLoader._parse_score(score_raw, label_map)
                    start_ms = int(start_raw) if start_raw else 0
                    end_ms = int(end_raw) if end_raw else 0
                    ranges.append(GTRange(start_ms=start_ms, end_ms=end_ms, score=score, description=desc))
                except Exception as e:
                    raise ValueError(f"Malformed GT file at row {row_idx}: {e}")

        return ranges

    @staticmethod
    def _load_json(filepath: str, label_map: dict[str, float] | None) -> list[GTRange]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON GT must be an array of objects")

        ranges: list[GTRange] = []
        for idx, item in enumerate(data):
            score = GTLoader._parse_score(item.get("score", 0.0), label_map)
            start_ms = int(item.get("start_ms", 0))
            end_ms = int(item.get("end_ms", 0))
            desc = item.get("description", "")
            ranges.append(GTRange(start_ms=start_ms, end_ms=end_ms, score=score, description=desc))
        return ranges
