from __future__ import annotations
"""
jev_gate.py - Pure JEV gate logic: tick schedule, routing, helpers.

Imports only: dataclasses, typing, math, app.core.types.
No cv2, torch, PySide6.
"""
import math
from dataclasses import dataclass, field
from typing import Sequence

from app.core.types import TickRecord, GateRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HAZARDS: tuple[str, ...] = (
    "pedestrian_vehicle",
    "no_ppe",
    "blocked_exit",
    "unsafe_load",
    "fall_or_spill",
)

TIER_FULL = "FULL"
TIER_REDUCED = "REDUCED"

REASON_INCOMPLETE = "jev_incomplete"
REASON_ABOVE_FULL = "above_full"
REASON_ABOVE_WATCH = "above_watch"
REASON_RISING = "rising"
REASON_HOLD = "hold"
REASON_CLEAR = "clear"

EPS = 1e-9


# ---------------------------------------------------------------------------
# GateConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateConfig:
    jev_window_s: float = 3.0
    jev_stride_s: float = 1.5
    jev_sample_fps: float = 4.0
    jev_max_side: int = 640
    route_full_threshold: float = 0.35
    route_watch_threshold: float = 0.20
    route_rise_delta: float = 0.10
    route_hold_chunks: int = 2
    route_reduced_fps: float = 1.0
    route_watch_exit_threshold: float | None = None   # hysteresis exit; None = disabled
    route_min_consecutive_watch: int = 0              # 0 or 1 = fire immediately (v2 behaviour)
    ewma_alpha: float | None = None                   # None = EWMA off
    prompt_hash: str = ""                             # Phase C; flows into GateLog via to_dict()
    hazard_full_thresholds: dict[str, float] | None = None
    hazard_watch_thresholds: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.jev_window_s <= 0:
            raise ValueError("jev_window_s must be > 0")
        if self.jev_stride_s <= 0:
            raise ValueError("jev_stride_s must be > 0")
        if self.jev_stride_s > self.jev_window_s:
            raise ValueError("jev_stride_s must be <= jev_window_s")
        if self.jev_sample_fps <= 0:
            raise ValueError("jev_sample_fps must be > 0")
        if not isinstance(self.jev_max_side, int) or self.jev_max_side < 32:
            raise ValueError("jev_max_side must be an int >= 32")
        if not (0 < self.route_watch_threshold <= self.route_full_threshold <= 1):
            raise ValueError(
                "must hold: 0 < route_watch_threshold <= route_full_threshold <= 1"
            )
        if not (0 < self.route_rise_delta <= 1):
            raise ValueError("route_rise_delta must be in (0, 1]")
        if not isinstance(self.route_hold_chunks, int) or isinstance(self.route_hold_chunks, bool) or self.route_hold_chunks < 0:
            raise ValueError("route_hold_chunks must be a non-negative int")
        if self.route_reduced_fps <= 0:
            raise ValueError("route_reduced_fps must be > 0")
        if self.route_watch_exit_threshold is not None and not (
                0 < self.route_watch_exit_threshold <= self.route_watch_threshold):
            raise ValueError(
                "route_watch_exit_threshold must satisfy 0 < route_watch_exit_threshold <= route_watch_threshold"
            )
        if (not isinstance(self.route_min_consecutive_watch, int)
                or isinstance(self.route_min_consecutive_watch, bool)
                or self.route_min_consecutive_watch < 0):
            raise ValueError("route_min_consecutive_watch must be a non-negative int")
        if self.ewma_alpha is not None and not (0 < self.ewma_alpha <= 1):
            raise ValueError("ewma_alpha must be in (0, 1] when set")
        if self.hazard_full_thresholds is not None:
            for h, val in self.hazard_full_thresholds.items():
                if h not in HAZARDS:
                    raise ValueError(f"unknown hazard in hazard_full_thresholds: {h}")
                if not (0 < val <= 1):
                    raise ValueError(f"hazard_full_thresholds[{h}] must be in (0, 1]")
        if self.hazard_watch_thresholds is not None:
            for h, val in self.hazard_watch_thresholds.items():
                if h not in HAZARDS:
                    raise ValueError(f"unknown hazard in hazard_watch_thresholds: {h}")
                if not (0 < val <= 1):
                    raise ValueError(f"hazard_watch_thresholds[{h}] must be in (0, 1]")
                full_val = self.get_full_threshold(h)
                if val > full_val:
                    raise ValueError(
                        f"hazard_watch_thresholds[{h}] ({val}) must be <= full threshold ({full_val})"
                    )
        if self.window_ms < 1:
            raise ValueError("window_ms must be >= 1")
        if self.stride_ms < 1:
            raise ValueError("stride_ms must be >= 1")

    def get_full_threshold(self, hazard: str) -> float:
        if self.hazard_full_thresholds and hazard in self.hazard_full_thresholds:
            return float(self.hazard_full_thresholds[hazard])
        return self.route_full_threshold

    def get_watch_threshold(self, hazard: str) -> float:
        if self.hazard_watch_thresholds and hazard in self.hazard_watch_thresholds:
            return float(self.hazard_watch_thresholds[hazard])
        return self.route_watch_threshold

    @property
    def window_ms(self) -> int:
        return int(round(self.jev_window_s * 1000))

    @property
    def stride_ms(self) -> int:
        return int(round(self.jev_stride_s * 1000))

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


def validate_against_vlm_fps(cfg: GateConfig, vlm_fps: float) -> None:
    if cfg.route_reduced_fps >= vlm_fps:
        raise ValueError(
            f"route_reduced_fps ({cfg.route_reduced_fps}) must be lower than vlm_fps ({vlm_fps})"
        )


# ---------------------------------------------------------------------------
# §5.1 Tick schedule
# ---------------------------------------------------------------------------

def compute_tick_times(duration_ms: int, window_ms: int, stride_ms: int) -> list[int]:
    if window_ms < 1:
        raise ValueError(f"window_ms must be >= 1, got {window_ms}")
    if stride_ms < 1:
        raise ValueError(f"stride_ms must be >= 1, got {stride_ms}")
    if stride_ms > window_ms:
        raise ValueError(f"stride_ms ({stride_ms}) must be <= window_ms ({window_ms})")
    if duration_ms <= 0:
        return []

    ticks: list[int] = []
    j = 0
    while True:
        t = window_ms + j * stride_ms
        if t > duration_ms:
            break
        ticks.append(t)
        j += 1

    if not ticks or ticks[-1] < duration_ms:
        ticks.append(duration_ms)

    return ticks


# ---------------------------------------------------------------------------
# §5.3 Tick-to-chunk assignment
# ---------------------------------------------------------------------------

def assign_ticks_to_chunks(
    tick_times: list[int],
    windows: list[tuple[int, int]],
) -> list[list[int]]:
    if not windows:
        return []

    result: list[list[int]] = [[] for _ in windows]
    n = len(windows)

    for tick_idx, t in enumerate(tick_times):
        placed = False
        for chunk_idx in range(n):
            e_prev = windows[chunk_idx - 1][1] if chunk_idx > 0 else -1
            e_k = windows[chunk_idx][1]
            if e_prev < t <= e_k:
                result[chunk_idx].append(tick_idx)
                placed = True
                break
        if not placed:
            # Beyond the last window's end -> last chunk
            result[n - 1].append(tick_idx)

    return result


# ---------------------------------------------------------------------------
# §5.4 Routing — GateRouter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteDecision:
    tier: str
    reason: str
    vec: dict[str, float] | None
    pmax: float | None
    rising_hazard: str | None
    hold_remaining_after: int
    ewma: dict[str, float] | None = None


class GateRouter:
    def __init__(self, cfg: GateConfig) -> None:
        self._cfg = cfg
        self.prev_vec: dict[str, float] | None = None
        self.hold_remaining: int = 0
        self.elevated: bool = False
        self.watch_streak: int = 0
        self.ewma: dict[str, float] | None = None

    def _update_ewma(self, vec: dict[str, float]) -> None:
        alpha = self._cfg.ewma_alpha
        if alpha is None:
            return
        if self.ewma is None:
            self.ewma = dict(vec)
        else:
            self.ewma = {h: alpha * vec[h] + (1.0 - alpha) * self.ewma[h] for h in HAZARDS}

    def decide(self, ticks: Sequence[TickRecord]) -> RouteDecision:
        cfg = self._cfg

        def ge(x: float, thr: float) -> bool:
            return x >= thr - EPS

        # Rule 1: empty or any error tick
        if not ticks or any(t.status != "ok" for t in ticks):
            self.prev_vec = None
            self.watch_streak = 0
            return RouteDecision(
                tier=TIER_FULL,
                reason=REASON_INCOMPLETE,
                vec=None,
                pmax=None,
                rising_hazard=None,
                hold_remaining_after=self.hold_remaining,
                ewma=self.ewma,
            )

        # Compute vec and pmax
        vec: dict[str, float] = {
            h: max(t.probs[h] for t in ticks)  # type: ignore[index]
            for h in HAZARDS
        }
        pmax = max(vec.values())
        full_now = any(ge(vec[h], cfg.get_full_threshold(h)) for h in HAZARDS)
        watch_now = any(ge(vec[h], cfg.get_watch_threshold(h)) for h in HAZARDS)

        # Rule 2: above_full
        if full_now:
            self.hold_remaining = cfg.route_hold_chunks
            self.prev_vec = dict(vec)
            self.elevated = True
            self.watch_streak = 0
            self._update_ewma(vec)
            return RouteDecision(
                tier=TIER_FULL,
                reason=REASON_ABOVE_FULL,
                vec=vec,
                pmax=pmax,
                rising_hazard=None,
                hold_remaining_after=self.hold_remaining,
                ewma=self.ewma,
            )

        # Rule 3: above_watch (with the v2-min_consecutive_watch streak condition)
        if watch_now and (cfg.route_min_consecutive_watch <= 1
                          or self.watch_streak + 1 >= cfg.route_min_consecutive_watch):
            self.elevated = True
            self.watch_streak += 1
            self.prev_vec = dict(vec)
            self._update_ewma(vec)
            return RouteDecision(
                tier=TIER_FULL,
                reason=REASON_ABOVE_WATCH,
                vec=vec,
                pmax=pmax,
                rising_hazard=None,
                hold_remaining_after=self.hold_remaining,
                ewma=self.ewma,
            )

        # Rule 3b: hysteresis exit — stay FULL while elevated and pmax >= exit
        if (cfg.route_watch_exit_threshold is not None and self.elevated
                and ge(pmax, cfg.route_watch_exit_threshold)):
            self.prev_vec = dict(vec)
            self.watch_streak = 0
            self._update_ewma(vec)
            return RouteDecision(
                tier=TIER_FULL,
                reason=REASON_ABOVE_WATCH,
                vec=vec,
                pmax=pmax,
                rising_hazard=None,
                hold_remaining_after=self.hold_remaining,
                ewma=self.ewma,
            )

        # Rule 4: rising
        rising_hazard: str | None = None
        if self.prev_vec is not None:
            for h in HAZARDS:
                if ge(vec[h] - self.prev_vec[h], cfg.route_rise_delta):
                    rising_hazard = h
                    break
        if rising_hazard is not None:
            self.elevated = True
            self.watch_streak = self.watch_streak + 1 if watch_now else 0
            self.prev_vec = dict(vec)
            self._update_ewma(vec)
            return RouteDecision(
                tier=TIER_FULL,
                reason=REASON_RISING,
                vec=vec,
                pmax=pmax,
                rising_hazard=rising_hazard,
                hold_remaining_after=self.hold_remaining,
                ewma=self.ewma,
            )

        # Rule 5: hold
        if self.hold_remaining > 0:
            self.hold_remaining -= 1
            self.watch_streak = self.watch_streak + 1 if watch_now else 0
            self.prev_vec = dict(vec)
            self._update_ewma(vec)
            return RouteDecision(
                tier=TIER_FULL,
                reason=REASON_HOLD,
                vec=vec,
                pmax=pmax,
                rising_hazard=None,
                hold_remaining_after=self.hold_remaining,
                ewma=self.ewma,
            )

        # Rule 6: clear
        self.elevated = False
        self.watch_streak = 0
        self.prev_vec = dict(vec)
        self._update_ewma(vec)
        return RouteDecision(
            tier=TIER_REDUCED,
            reason=REASON_CLEAR,
            vec=vec,
            pmax=pmax,
            rising_hazard=None,
            hold_remaining_after=self.hold_remaining,
            ewma=self.ewma,
        )


# ---------------------------------------------------------------------------
# §5.6 Summary and formatting helpers
# ---------------------------------------------------------------------------

def summarize_gate(records: Sequence[GateRecord]) -> dict:
    n_chunks = len(records)
    n_full = sum(1 for r in records if r.tier == TIER_FULL)
    n_reduced = n_chunks - n_full
    frames_sent = sum(r.frames_sent for r in records)
    frames_full_equiv = sum(r.frames_full_equiv for r in records)
    frames_saved_pct: float | None = (
        (frames_full_equiv - frames_sent) / frames_full_equiv * 100
        if frames_full_equiv > 0
        else None
    )
    jev_time_s = sum(r.jev_time_s for r in records)
    vlm_time_s = sum(r.vlm_time_s for r in records if r.vlm_time_s is not None)
    reasons: dict[str, int] = {}
    for r in records:
        reasons[r.reason] = reasons.get(r.reason, 0) + 1
    return {
        "n_chunks": n_chunks,
        "n_full": n_full,
        "n_reduced": n_reduced,
        "frames_sent": frames_sent,
        "frames_full_equiv": frames_full_equiv,
        "frames_saved_pct": frames_saved_pct,
        "jev_time_s": jev_time_s,
        "vlm_time_s": vlm_time_s,
        "reasons": reasons,
    }


def format_time_range(start_ms: int, end_ms: int) -> str:
    def _fmt(ms: int) -> str:
        total_s = ms // 1000
        m = total_s // 60
        s = total_s % 60
        return f"{m}:{s:02d}"
    return f"{_fmt(start_ms)}-{_fmt(end_ms)}"


def format_tick_line(tick: TickRecord) -> str:
    if tick.status != "ok" or tick.probs is None:
        return f"t={tick.t_ms}ms error error={tick.error}"
    top_name = max(HAZARDS, key=lambda h: tick.probs[h])  # type: ignore[index]
    top_p = tick.probs[top_name]  # type: ignore[index]
    rest = " ".join(f"{h}={tick.probs[h]:.2f}" for h in HAZARDS)  # type: ignore[index]
    return f"t={tick.t_ms}ms ok top={top_name}:{top_p:.2f} [{rest}]"


def tick_to_dict(tick: TickRecord) -> dict:
    return {
        "tick_id": tick.tick_id,
        "t_ms": tick.t_ms,
        "window_start_ms": tick.window_start_ms,
        "num_frames": tick.num_frames,
        "status": tick.status,
        "probs": tick.probs,
        "error": tick.error,
        "jev_time_s": tick.jev_time_s,
    }


def gate_record_to_dict(rec: GateRecord, include_ticks: bool) -> dict:
    d: dict = {
        "chunk_id": rec.chunk_id,
        "start_ms": rec.start_ms,
        "end_ms": rec.end_ms,
        "tier": rec.tier,
        "reason": rec.reason,
        "vec": rec.vec,
        "pmax": rec.pmax,
        "rising_hazard": rec.rising_hazard,
        "tick_ids": rec.tick_ids,
        "num_ticks_ok": rec.num_ticks_ok,
        "num_ticks_error": rec.num_ticks_error,
        "sample_fps": rec.sample_fps,
        "frames_sent": rec.frames_sent,
        "frames_full_equiv": rec.frames_full_equiv,
        "jev_time_s": rec.jev_time_s,
        "vlm_time_s": rec.vlm_time_s,
        "vlm_status": rec.vlm_status,
        "vlm_score": rec.vlm_score,
        "wall_done_s": rec.wall_done_s,
        "hold_remaining_after": rec.hold_remaining_after,
        "top_tick_id": rec.top_tick_id,
        "ewma": rec.ewma,
    }
    if include_ticks:
        d["ticks"] = [tick_to_dict(t) for t in rec.ticks]
    return d