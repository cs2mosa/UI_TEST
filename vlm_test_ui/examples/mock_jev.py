from __future__ import annotations
"""
mock_jev.py - MockJevDecider for tests and demo runs (§7.3, §12 MOCK-1..4).

No torch, no cv2, no network. Pure Python + numpy.
"""
import json
import os
import time

import numpy as np

from app.core.jev_gate import HAZARDS

# ---------------------------------------------------------------------------
# Default demo rules (§7.3)
# Rule = (start_ms, end_ms, {hazard: p}); first match wins.
# ---------------------------------------------------------------------------
DEMO_RULES: list[tuple[int, int, dict[str, float]]] = [
    (5500, 8000, {"no_ppe": 0.25}),
    (10000, 12000, {"no_ppe": 0.60}),
]


class MockJevDecider:
    """
    Deterministic JEV decider for tests.

    Output depends only on t_ms (rules + default); frames_rgb is accepted but
    not inspected (D-17).
    """

    label = "mock"

    def __init__(
        self,
        rules: list[tuple[int, int, dict[str, float]]] | None = None,
        default: float = 0.05,
        delay_s: float = 0.0,
        fail_at: tuple[int, ...] = (),
        bad_output_at: tuple[int, ...] = (),
    ) -> None:
        self._rules: list[tuple[int, int, dict[str, float]]] = (
            DEMO_RULES if rules is None else list(rules)
        )
        self._default = default
        self._delay_s = delay_s
        self._fail_at: set[int] = set(fail_at)
        self._bad_output_at: set[int] = set(bad_output_at)

    def __call__(
        self,
        frames_rgb: list[np.ndarray],
        fps: float,
        t_ms: int,
    ) -> dict[str, float]:
        if self._delay_s > 0:
            time.sleep(self._delay_s)
        if t_ms in self._fail_at:
            raise RuntimeError(f"mock failure at t={t_ms}")

        # Build probs: default for all, then overlay the first matching rule
        probs = {h: self._default for h in HAZARDS}
        for start_ms, end_ms, overrides in self._rules:
            if start_ms <= t_ms <= end_ms:
                for h, p in overrides.items():
                    if h in probs:
                        probs[h] = p
                break   # first rule wins

        if t_ms in self._bad_output_at:
            del probs["blocked_exit"]

        return probs

    @classmethod
    def from_json(cls, path: str) -> "MockJevDecider":
        """
        Load from a JSON file.  Schema:
            {
              "default": float,          # optional
              "delay_s": float,          # optional
              "rules": [[start, end, {hazard: p}], ...],  # optional
              "fail_at": [int, ...],     # optional
              "bad_output_at": [int, ...]  # optional
            }
        Unknown keys -> ValueError.
        """
        allowed_keys = {"default", "delay_s", "rules", "fail_at", "bad_output_at"}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"mock_jev JSON must be an object, got {type(data).__name__}")
        unknown = set(data.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown keys in mock_jev JSON: {sorted(unknown)}")

        rules_raw = data.get("rules", None)
        rules: list[tuple[int, int, dict[str, float]]] | None = None
        if rules_raw is not None:
            rules = [(int(r[0]), int(r[1]), dict(r[2])) for r in rules_raw]

        return cls(
            rules=rules,
            default=float(data.get("default", 0.05)),
            delay_s=float(data.get("delay_s", 0.0)),
            fail_at=tuple(int(x) for x in data.get("fail_at", [])),
            bad_output_at=tuple(int(x) for x in data.get("bad_output_at", [])),
        )