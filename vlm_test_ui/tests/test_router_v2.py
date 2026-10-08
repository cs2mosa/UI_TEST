from __future__ import annotations
"""
test_router_v2.py - Phase D ROUTE2-1..8 + LOGX-1 tests (design_v3 §5.5, §10.1).

Each new knob fires; each is inert at defaults; the frozen-v2 identity property holds
over >= 1000 randomized tick sequences (seeded, mixed ok/error statuses) against an
embedded copy of the v2 router; EWMA is log-only; the GateRecord dict gains exactly
one trailing "ewma" key after "top_tick_id".

CPU-only; no GPU, no network. The worker part of LOGX-1 runs offscreen with the
MockJevDecider, exactly as test_gated_worker.py drives it.
"""
import json
import os
import random
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.core.jev_gate import (
    EPS,
    GateConfig,
    GateRouter,
    HAZARDS,
    REASON_ABOVE_FULL,
    REASON_ABOVE_WATCH,
    REASON_CLEAR,
    REASON_HOLD,
    REASON_INCOMPLETE,
    REASON_RISING,
    TIER_FULL,
    TIER_REDUCED,
    gate_record_to_dict,
)
from app.core.types import TickRecord


def _ok_tick(tick_id, t_ms, probs):
    return TickRecord(
        tick_id=tick_id, t_ms=t_ms, window_start_ms=max(0, t_ms - 3000),
        num_frames=4, status="ok",
        probs={h: probs.get(h, 0.0) for h in HAZARDS},
    )


def _err_tick(tick_id, t_ms):
    return TickRecord(
        tick_id=tick_id, t_ms=t_ms, window_start_ms=max(0, t_ms - 3000),
        num_frames=0, status="error", error="no_frames",
    )


def _flat(p):
    return {h: p for h in HAZARDS}


# ---------------------------------------------------------------------------
# Frozen copy of the v2 GateRouter.decide (the identity oracle; never edited)
# ---------------------------------------------------------------------------

class _V2OracleRouter:
    """Verbatim v2 GateRouter (design_v2 §5.4) at the D-29-frozen point."""

    def __init__(self, cfg: GateConfig) -> None:
        self._cfg = cfg
        self.prev_vec = None
        self.hold_remaining = 0

    def decide(self, ticks):
        cfg = self._cfg

        def ge(x, thr):
            return x >= thr - EPS

        if not ticks or any(t.status != "ok" for t in ticks):
            self.prev_vec = None
            return dict(tier=TIER_FULL, reason=REASON_INCOMPLETE, vec=None, pmax=None,
                        rising_hazard=None, hold_remaining_after=self.hold_remaining)

        vec = {h: max(t.probs[h] for t in ticks) for h in HAZARDS}
        pmax = max(vec.values())

        if ge(pmax, cfg.route_full_threshold):
            self.hold_remaining = cfg.route_hold_chunks
            self.prev_vec = dict(vec)
            return dict(tier=TIER_FULL, reason=REASON_ABOVE_FULL, vec=vec, pmax=pmax,
                        rising_hazard=None, hold_remaining_after=self.hold_remaining)

        if ge(pmax, cfg.route_watch_threshold):
            self.prev_vec = dict(vec)
            return dict(tier=TIER_FULL, reason=REASON_ABOVE_WATCH, vec=vec, pmax=pmax,
                        rising_hazard=None, hold_remaining_after=self.hold_remaining)

        rising_hazard = None
        if self.prev_vec is not None:
            for h in HAZARDS:
                if ge(vec[h] - self.prev_vec[h], cfg.route_rise_delta):
                    rising_hazard = h
                    break
        if rising_hazard is not None:
            self.prev_vec = dict(vec)
            return dict(tier=TIER_FULL, reason=REASON_RISING, vec=vec, pmax=pmax,
                        rising_hazard=rising_hazard, hold_remaining_after=self.hold_remaining)

        if self.hold_remaining > 0:
            self.hold_remaining -= 1
            self.prev_vec = dict(vec)
            return dict(tier=TIER_FULL, reason=REASON_HOLD, vec=vec, pmax=pmax,
                        rising_hazard=None, hold_remaining_after=self.hold_remaining)

        self.prev_vec = dict(vec)
        return dict(tier=TIER_REDUCED, reason=REASON_CLEAR, vec=vec, pmax=pmax,
                    rising_hazard=None, hold_remaining_after=self.hold_remaining)


def _assert_v2_shape(d):
    return (d.tier, d.reason, d.vec, d.pmax, d.rising_hazard, d.hold_remaining_after)


class TestRoute2KnobsFire(unittest.TestCase):
    # ROUTE2-1a: route_min_consecutive_watch requires a streak; the streak builds
    # through rules 4/5 (rule 6 resets it), so the sequence must rise into watch band
    def test_min_consecutive_watch_fires_on_streak(self):
        cfg = GateConfig(route_full_threshold=0.50, route_watch_threshold=0.30,
                         route_min_consecutive_watch=2)
        r = GateRouter(cfg)
        d1 = r.decide([_ok_tick(0, 3000, _flat(0.05))])   # clear
        self.assertEqual((d1.tier, d1.reason), (TIER_REDUCED, REASON_CLEAR))
        d2 = r.decide([_ok_tick(1, 6000, _flat(0.35))])   # rising (delta 0.30) -> streak 1
        self.assertEqual((d2.tier, d2.reason), (TIER_FULL, REASON_RISING))
        d3 = r.decide([_ok_tick(2, 9000, _flat(0.35))])   # streak 1+1 >= 2 -> rule 3 fires
        self.assertEqual((d3.tier, d3.reason), (TIER_FULL, REASON_ABOVE_WATCH))

    # ROUTE2-1b: hysteresis exit keeps elevated chunks FULL below watch
    def test_watch_exit_keeps_elevated_full(self):
        cfg = GateConfig(route_full_threshold=0.90, route_watch_threshold=0.50,
                         route_watch_exit_threshold=0.30)
        r = GateRouter(cfg)
        r.decide([_ok_tick(0, 3000, _flat(0.60))])          # above_watch -> elevated
        d = r.decide([_ok_tick(1, 6000, _flat(0.35))])      # below watch, above exit
        self.assertEqual((d.tier, d.reason), (TIER_FULL, REASON_ABOVE_WATCH))
        r2 = GateRouter(GateConfig(route_full_threshold=0.90, route_watch_threshold=0.50))
        r2.decide([_ok_tick(0, 3000, _flat(0.60))])
        d2 = r2.decide([_ok_tick(1, 6000, _flat(0.35))])
        self.assertEqual((d2.tier, d2.reason), (TIER_REDUCED, REASON_CLEAR))

    # ROUTE2-1c: ewma_alpha produces the log-only EWMA
    def test_ewma_fires(self):
        cfg = GateConfig(ewma_alpha=0.5)
        r = GateRouter(cfg)
        d1 = r.decide([_ok_tick(0, 3000, _flat(0.2))])
        self.assertEqual(d1.ewma, _flat(0.2))
        d2 = r.decide([_ok_tick(1, 6000, _flat(0.6))])
        self.assertEqual(d2.ewma, _flat(0.4))


class TestRoute2DefaultsInert(unittest.TestCase):
    # ROUTE2-2: every knob inert at defaults (exit off, streak 0, ewma off)
    def test_defaults_ewma_none_and_3b_never_fires(self):
        cfg = GateConfig()
        r = GateRouter(cfg)
        seqs = [_flat(0.6), _flat(0.3), _flat(0.25), _flat(0.05), _flat(0.21)]
        for i, p in enumerate(seqs):
            d = r.decide([_ok_tick(i, 3000 * (i + 1), p)])
            self.assertIsNone(d.ewma)
        # a watch-level chunk then a below-watch chunk: without exit -> clear
        r2 = GateRouter(GateConfig(route_full_threshold=0.9, route_watch_threshold=0.5))
        r2.decide([_ok_tick(0, 3000, _flat(0.6))])
        d2 = r2.decide([_ok_tick(1, 6000, _flat(0.49))])
        self.assertEqual((d2.tier, d2.reason), (TIER_REDUCED, REASON_CLEAR))

    def test_validation_of_new_fields(self):
        with self.assertRaises(ValueError):
            GateConfig(route_full_threshold=0.95, route_watch_threshold=0.5, route_watch_exit_threshold=0.9)
        with self.assertRaises(ValueError):
            GateConfig(route_full_threshold=0.95, route_watch_threshold=0.5, route_watch_exit_threshold=0.0)
        with self.assertRaises(ValueError):
            GateConfig(route_min_consecutive_watch=-1)
        with self.assertRaises(ValueError):
            GateConfig(route_min_consecutive_watch=True)
        with self.assertRaises(ValueError):
            GateConfig(ewma_alpha=0.0)
        with self.assertRaises(ValueError):
            GateConfig(ewma_alpha=1.5)
        GateConfig(route_full_threshold=0.95, route_watch_threshold=0.5,
                   route_watch_exit_threshold=0.5)  # exit == watch allowed
        GateConfig(route_min_consecutive_watch=0, ewma_alpha=1.0, prompt_hash="abc")


class TestRoute2FrozenV2Identity(unittest.TestCase):
    # ROUTE2-3: >= 1000 randomized tick sequences; defaults identical to the v2 oracle
    def test_frozen_v2_identity_over_random_sequences(self):
        rng = random.Random(20261007)
        for trial in range(1000):
            n_chunks = rng.randint(1, 6)
            oracle = _V2OracleRouter(GateConfig())
            v3 = GateRouter(GateConfig())
            for c in range(n_chunks):
                ticks = []
                n_ticks = rng.randint(0, 3)
                for k in range(n_ticks):
                    t_ms = 3000 * (c + 1) + k
                    if rng.random() < 0.15:
                        ticks.append(_err_tick(k, t_ms))
                    else:
                        probs = {h: rng.random() for h in HAZARDS}
                        ticks.append(_ok_tick(k, t_ms, probs))
                d3 = v3.decide(ticks)
                o = oracle.decide(ticks)
                self.assertEqual(_assert_v2_shape(d3), (o["tier"], o["reason"], o["vec"],
                                                        o["pmax"], o["rising_hazard"],
                                                        o["hold_remaining_after"]),
                                 msg=f"trial={trial} chunk={c} ticks={n_ticks}")
                self.assertIsNone(d3.ewma)


class TestRoute2StateDetails(unittest.TestCase):
    # ROUTE2-4: EWMA arithmetic across chunks, per hazard
    def test_ewma_per_hazard(self):
        cfg = GateConfig(ewma_alpha=0.25)
        r = GateRouter(cfg)
        probs1 = {h: 0.0 for h in HAZARDS}
        probs2 = {h: 0.0 for h in HAZARDS}
        probs1["no_ppe"] = 0.4
        probs2["blocked_exit"] = 0.8
        d1 = r.decide([_ok_tick(0, 3000, probs1)])
        d2 = r.decide([_ok_tick(1, 6000, probs2)])
        self.assertAlmostEqual(d2.ewma["no_ppe"], 0.25 * 0.0 + 0.75 * 0.4, places=12)
        self.assertAlmostEqual(d2.ewma["blocked_exit"], 0.25 * 0.8, places=12)

    # ROUTE2-5: streak resets on rule 1 (error tick)
    def test_streak_reset_on_rule_1(self):
        cfg = GateConfig(route_full_threshold=0.50, route_watch_threshold=0.30,
                         route_min_consecutive_watch=3)
        r = GateRouter(cfg)
        r.decide([_ok_tick(0, 3000, _flat(0.35))])   # clear (streak 0 -> rule 3 needs 3)
        self.assertEqual(r.watch_streak, 0)
        r.decide([_ok_tick(1, 6000, _flat(0.35))])   # clear; streak stays 0 (rule 3 not fired)
        r.decide([_err_tick(2, 9000)])               # rule 1 -> streak reset (was already 0)
        d = r.decide([_ok_tick(3, 12000, _flat(0.35))])
        self.assertEqual((d.tier, d.reason), (TIER_REDUCED, REASON_CLEAR))
        self.assertEqual(r.watch_streak, 0)

    def test_streak_builds_then_resets_on_error(self):
        cfg = GateConfig(route_full_threshold=0.90, route_watch_threshold=0.10,
                         route_min_consecutive_watch=3)
        r = GateRouter(cfg)
        r.decide([_ok_tick(0, 3000, _flat(0.2))])    # rule 3 (min_consec<=1? no: streak 0+1<3) -> falls through
        # watch=0.10, pmax=0.2: rule 3 blocked by streak, rule 4/5 miss -> clear
        self.assertEqual(r.watch_streak, 0)
        r2 = GateRouter(cfg)
        r2.decide([_ok_tick(0, 3000, _flat(0.2))])
        r2.decide([_ok_tick(1, 6000, _flat(0.2))])   # streak 1+1 >= 3? no -> clear
        self.assertEqual(r2.watch_streak, 0)
        r3 = GateRouter(cfg)
        r3.decide([_ok_tick(0, 3000, _flat(0.2))])
        r3.decide([_err_tick(1, 6000)])              # rule 1 resets whatever streak existed
        self.assertEqual(r3.watch_streak, 0)
        d = r3.decide([_ok_tick(2, 9000, _flat(0.2))])
        self.assertEqual(d.reason, REASON_CLEAR)

    # ROUTE2-6: hysteresis boundary uses ge EPS semantics
    def test_3b_boundary_ge_eps(self):
        cfg = GateConfig(route_full_threshold=0.90, route_watch_threshold=0.50,
                         route_watch_exit_threshold=0.30)
        r = GateRouter(cfg)
        r.decide([_ok_tick(0, 3000, _flat(0.60))])           # elevated
        d = r.decide([_ok_tick(1, 6000, _flat(0.30 - 2 * EPS))])
        self.assertEqual((d.tier, d.reason), (TIER_REDUCED, REASON_CLEAR))
        r2 = GateRouter(cfg)
        r2.decide([_ok_tick(0, 3000, _flat(0.60))])
        d2 = r2.decide([_ok_tick(1, 6000, _flat(0.30 + 0.0))])  # exactly exit -> ge fires
        self.assertEqual(d2.reason, REASON_ABOVE_WATCH)

    # ROUTE2-7: hold_remaining_after correctness with 3b present
    def test_hold_with_3b(self):
        cfg = GateConfig(route_full_threshold=0.50, route_watch_threshold=0.20,
                         route_watch_exit_threshold=0.05, route_hold_chunks=2)
        r = GateRouter(cfg)
        d0 = r.decide([_ok_tick(0, 3000, _flat(0.60))])      # above_full, hold=2
        self.assertEqual(d0.hold_remaining_after, 2)
        d1 = r.decide([_ok_tick(1, 6000, _flat(0.10))])      # below watch, >= exit, elevated -> 3b
        self.assertEqual(d1.reason, REASON_ABOVE_WATCH)
        self.assertEqual(d1.hold_remaining_after, 2)         # 3b does not consume hold
        d2 = r.decide([_ok_tick(2, 9000, _flat(0.01))])      # below exit -> rule 5 hold
        self.assertEqual(d2.reason, REASON_HOLD)
        self.assertEqual(d2.hold_remaining_after, 1)


class TestRoute2Logx(unittest.TestCase):
    # LOGX-1a: gate_record_to_dict key order = v2 order + trailing "ewma"
    def test_logx_1a_dict_key_order(self):
        from app.core.types import GateRecord
        tick = TickRecord(tick_id=0, t_ms=3000, window_start_ms=0, num_frames=4,
                          status="ok", probs=_flat(0.05), jev_time_s=0.01)
        rec = GateRecord(
            chunk_id=0, start_ms=0, end_ms=5000, tier="FULL", reason="above_watch",
            vec=_flat(0.05), pmax=0.05, rising_hazard=None, tick_ids=[0], ticks=[tick],
            top_tick_id=0, num_ticks_ok=1, num_ticks_error=0, sample_fps=4.0,
            frames_sent=20, frames_full_equiv=20, jev_time_s=0.01, vlm_time_s=None,
            vlm_status=None, vlm_score=None, wall_done_s=1.0, hold_remaining_after=0,
            ewma=_flat(0.05),
        )
        d = gate_record_to_dict(rec, include_ticks=False)
        expected_keys = [
            "chunk_id", "start_ms", "end_ms", "tier", "reason",
            "vec", "pmax", "rising_hazard", "tick_ids",
            "num_ticks_ok", "num_ticks_error", "sample_fps",
            "frames_sent", "frames_full_equiv", "jev_time_s",
            "vlm_time_s", "vlm_status", "vlm_score", "wall_done_s",
            "hold_remaining_after", "top_tick_id",
            "ewma",
        ]
        self.assertEqual(list(d.keys()), expected_keys)
        json.dumps(d, separators=(",", ":"), allow_nan=False)  # serialisable

    # LOGX-1b: GateConfig.to_dict() contains prompt_hash
    def test_logx_1b_to_dict_prompt_hash(self):
        cfg = GateConfig(prompt_hash="deadbeef")
        self.assertEqual(cfg.to_dict()["prompt_hash"], "deadbeef")

    # LOGX-1c: worker-emitted GateRecord.ewma is None at default config (mock run)
    def test_logx_1c_worker_ewma_none_at_defaults(self):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv[:1])
        from app.workers.gated_batch_worker import GatedBatchWorker
        from examples.mock_jev import MockJevDecider
        from tests.support import make_test_video, RecordingAdapter
        video = make_test_video(duration_ms=12000, fps=30.0)
        log_dir = tempfile.mkdtemp()
        try:
            worker = GatedBatchWorker(
                media_path=video, adapter=RecordingAdapter(),
                chunk_s=5.0, overlap_s=1.0, vlm_fps=4.0, parent=None,
                decider=MockJevDecider(), gate_config=GateConfig(), log_dir=log_dir,
            )
            records, outcome = [], []
            worker.gate_ready.connect(records.append)
            worker.finished.connect(lambda: outcome.append("finished"))
            worker.error_occurred.connect(lambda m: outcome.append("error:" + m))
            worker.start()
            spins = 0
            while not outcome and spins < 60000:
                app.processEvents()
                spins += 1
            worker.wait()
            app.processEvents()
            self.assertTrue(outcome)
            self.assertTrue(all(o == "finished" for o in outcome),
                            msg=f"unexpected outcomes: {outcome}")  # finished may arrive twice (native QThread teardown)
            self.assertGreater(len(records), 0)
            for rec in records:
                self.assertIsNone(rec.ewma)
        finally:
            os.unlink(video)
            shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
