from __future__ import annotations
import unittest

from app.core.jev_gate import (
    GateConfig,
    GateRouter,
    compute_tick_times,
    assign_ticks_to_chunks,
    HAZARDS,
    TIER_FULL,
    TIER_REDUCED,
    REASON_INCOMPLETE,
    REASON_ABOVE_FULL,
    REASON_ABOVE_WATCH,
    REASON_RISING,
    REASON_HOLD,
    REASON_CLEAR,
    EPS,
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


class TestComputeTickTimes(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(compute_tick_times(12000, 3000, 1500), [3000, 4500, 6000, 7500, 9000, 10500, 12000])
        self.assertEqual(compute_tick_times(4000, 3000, 1500), [3000, 4000])
        self.assertEqual(compute_tick_times(2000, 3000, 1500), [2000])
        self.assertEqual(compute_tick_times(3000, 3000, 1500), [3000])
    def test_value_errors(self):
        with self.assertRaises(ValueError): compute_tick_times(10000, 0, 1500)
        with self.assertRaises(ValueError): compute_tick_times(10000, 3000, 0)
        with self.assertRaises(ValueError): compute_tick_times(10000, 1000, 2000)
    def test_zero_duration(self):
        self.assertEqual(compute_tick_times(0, 3000, 1500), [])
        self.assertEqual(compute_tick_times(-1, 3000, 1500), [])
    def test_forced_append(self):
        result = compute_tick_times(4000, 3000, 1500)
        self.assertEqual(result[-1], 4000)
    def test_strictly_ascending(self):
        result = compute_tick_times(12000, 3000, 1500)
        for a, b in zip(result, result[1:]): self.assertLess(a, b)
        self.assertEqual(result[-1], 12000)


class TestAssignTicksToChunks(unittest.TestCase):
    def test_boundary_exact(self):
        result = assign_ticks_to_chunks([5000], [(0, 5000), (4000, 9000)])
        self.assertEqual(result[0], [0]); self.assertEqual(result[1], [])
    def test_beyond_last_window(self):
        result = assign_ticks_to_chunks([5000, 11000], [(0, 5000), (4000, 9000)])
        self.assertIn(1, result[1])
    def test_empty_chunk(self):
        result = assign_ticks_to_chunks([5000], [(0, 5000), (4000, 9000), (8000, 12000)])
        self.assertEqual(result[0], [0]); self.assertEqual(result[1], []); self.assertEqual(result[2], [])
    def test_empty_windows(self):
        self.assertEqual(assign_ticks_to_chunks([1000, 2000], []), [])
    def test_every_tick_exactly_once(self):
        tts = [3000, 4500, 6000, 7500, 9000, 10500, 12000]
        result = assign_ticks_to_chunks(tts, [(0,5000),(4000,9000),(8000,12000)])
        all_i = [i for chunk in result for i in chunk]
        self.assertEqual(sorted(all_i), list(range(len(tts))))


class TestGateRouter(unittest.TestCase):
    def _r(self, **kw): return GateRouter(GateConfig(**kw))
    def test_rule1_empty(self):
        d = self._r().decide([]); self.assertEqual(d.tier, TIER_FULL); self.assertEqual(d.reason, REASON_INCOMPLETE); self.assertIsNone(d.vec)
    def test_rule1_error_tick(self):
        d = self._r().decide([_err_tick(0,3000)]); self.assertEqual(d.reason, REASON_INCOMPLETE)
    def test_rule2_above_full(self):
        d = self._r(route_full_threshold=0.35).decide([_ok_tick(0,3000,{"no_ppe":0.40})]); self.assertEqual(d.reason, REASON_ABOVE_FULL)
    def test_rule3_above_watch(self):
        d = self._r(route_full_threshold=0.35, route_watch_threshold=0.20).decide([_ok_tick(0,3000,{"no_ppe":0.25})]); self.assertEqual(d.reason, REASON_ABOVE_WATCH)
    def test_rule4_rising(self):
        # Use watch_threshold=0.30 so probs of 0.16 do not trigger above_watch (rule 3).
        r = self._r(route_rise_delta=0.10, route_watch_threshold=0.30, route_full_threshold=0.50)
        r.decide([_ok_tick(0,3000,_flat(0.05))])
        # no_ppe rises from 0.05 to 0.16 (delta 0.11 > 0.10), all below watch threshold 0.30
        d = r.decide([_ok_tick(1,6000,{"no_ppe":0.16,**{h:0.05 for h in HAZARDS if h!="no_ppe"}})])
        self.assertEqual(d.reason, REASON_RISING); self.assertEqual(d.rising_hazard, "no_ppe")
    def test_rule5_hold_countdown(self):
        r = self._r(route_full_threshold=0.35, route_hold_chunks=2)
        r.decide([_ok_tick(0,3000,{"no_ppe":0.40})])
        d1=r.decide([_ok_tick(1,6000,_flat(0.01))]); d2=r.decide([_ok_tick(2,9000,_flat(0.01))]); d3=r.decide([_ok_tick(3,12000,_flat(0.01))])
        self.assertEqual(d1.reason,REASON_HOLD); self.assertEqual(d2.reason,REASON_HOLD); self.assertEqual(d3.reason,REASON_CLEAR)
    def test_rule6_clear(self):
        d = self._r().decide([_ok_tick(0,3000,_flat(0.01))]); self.assertEqual(d.tier, TIER_REDUCED); self.assertEqual(d.reason, REASON_CLEAR)
    def test_eps_edge(self):
        # ge(x, thr) := x >= thr - EPS (EPS=1e-9). Value thr-2e-9 is strictly below thr-EPS.
        thr=0.35; r=self._r(route_full_threshold=thr, route_watch_threshold=0.20)
        d=r.decide([_ok_tick(0,3000,{"no_ppe":thr-2e-9})]); self.assertNotEqual(d.reason, REASON_ABOVE_FULL)
    def test_hold_reset_on_above_full(self):
        r=self._r(route_full_threshold=0.35, route_hold_chunks=3)
        r.decide([_ok_tick(0,3000,{"no_ppe":0.40})]); r.decide([_ok_tick(1,6000,_flat(0.01))])
        d=r.decide([_ok_tick(2,9000,{"no_ppe":0.40})]); self.assertEqual(d.reason,REASON_ABOVE_FULL); self.assertEqual(d.hold_remaining_after,3)
    def test_prev_vec_none_after_incomplete(self):
        r=self._r(route_rise_delta=0.10); r.decide([_ok_tick(0,3000,_flat(0.30))]); r.decide([_err_tick(1,6000)]); self.assertIsNone(r.prev_vec)
    def test_router_not_mutate_inputs(self):
        r=self._r(); ticks=[_ok_tick(0,3000,_flat(0.05))]; orig=dict(ticks[0].probs); r.decide(ticks); self.assertEqual(ticks[0].probs, orig)
    def test_hold_remaining_after_correct(self):
        r=self._r(route_full_threshold=0.35, route_hold_chunks=2)
        d=r.decide([_ok_tick(0,3000,{"no_ppe":0.40})]); self.assertEqual(d.hold_remaining_after,2)
        d2=r.decide([_ok_tick(1,6000,_flat(0.01))]); self.assertEqual(d2.hold_remaining_after,1)
    def test_rising_hazard_first_in_order(self):
        r=self._r(route_rise_delta=0.10, route_watch_threshold=0.30, route_full_threshold=0.50)
        r.decide([_ok_tick(0,3000,_flat(0.05))])
        h0,h1=HAZARDS[0],HAZARDS[1]; probs={h:0.05 for h in HAZARDS}; probs[h0]=0.16; probs[h1]=0.16
        d=r.decide([_ok_tick(1,6000,probs)]); self.assertEqual(d.rising_hazard, h0)
    def test_rule_precedence(self):
        r=self._r(route_full_threshold=0.35, route_watch_threshold=0.20)
        d=r.decide([_ok_tick(0,3000,{"no_ppe":0.50})]); self.assertEqual(d.reason, REASON_ABOVE_FULL)

if __name__ == "__main__": unittest.main()