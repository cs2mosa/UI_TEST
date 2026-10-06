from __future__ import annotations
import math
import sys
import tempfile
import os
import unittest
import time
import numpy as np
import cv2

from app.core.jev_adapter import validate_probs, evaluate_tick, HAZARD_QUESTIONS
from examples.mock_jev import MockJevDecider
from app.core.jev_gate import GateConfig, HAZARDS
from app.core.types import TickRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cap(n_frames=10, fps=30.0, width=64, height=64):
    tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), i * 20 % 255, dtype=np.uint8)
        out.write(frame)
    out.release()
    cap = cv2.VideoCapture(tmp.name)
    return cap, tmp.name


def _flat_decider(p=0.05, delay_s=0.0):
    """A simple callable decider returning flat probs."""
    class _Flat:
        called = 0
        def __call__(self, frames_rgb, fps, t_ms):
            self.called += 1
            if delay_s > 0:
                time.sleep(delay_s)
            return {h: p for h in HAZARDS}
    return _Flat()


class _RecordingDecider:
    """Records arguments for EVAL-2 proof."""
    called = 0
    def __call__(self, frames_rgb, fps, t_ms):
        self.called += 1
        return {h: 0.05 for h in HAZARDS}


class _ErrorDecider:
    def __call__(self, frames_rgb, fps, t_ms):
        raise ValueError("boom from decider")


class _BadOutputDecider:
    """Returns a dict missing a hazard key."""
    def __call__(self, frames_rgb, fps, t_ms):
        d = {h: 0.05 for h in HAZARDS}
        del d[HAZARDS[-1]]   # remove last hazard
        return d


class _KeyboardDecider:
    def __call__(self, frames_rgb, fps, t_ms):
        raise KeyboardInterrupt("SIGINT")


# ---------------------------------------------------------------------------
# VALID-1..5: validate_probs
# ---------------------------------------------------------------------------

class TestValidateProbs(unittest.TestCase):

    def test_valid_1_exact_key_set(self):
        raw = {h: 0.5 for h in HAZARDS}
        result = validate_probs(raw)
        self.assertEqual(set(result.keys()), set(HAZARDS))

    def test_valid_1_missing_key(self):
        raw = {h: 0.5 for h in HAZARDS}
        del raw[HAZARDS[0]]
        with self.assertRaises(ValueError) as ctx:
            validate_probs(raw)
        self.assertIn("invalid_output:", str(ctx.exception))

    def test_valid_1_extra_key(self):
        raw = {h: 0.5 for h in HAZARDS}
        raw["extra_hazard"] = 0.1
        with self.assertRaises(ValueError) as ctx:
            validate_probs(raw)
        self.assertIn("invalid_output:", str(ctx.exception))

    def test_valid_2_bool_rejected(self):
        raw = {h: 0.5 for h in HAZARDS}
        raw[HAZARDS[0]] = True
        with self.assertRaises(ValueError) as ctx:
            validate_probs(raw)
        self.assertIn("invalid_output:", str(ctx.exception))

    def test_valid_3_nan_rejected(self):
        raw = {h: 0.5 for h in HAZARDS}
        raw[HAZARDS[0]] = float("nan")
        with self.assertRaises(ValueError):
            validate_probs(raw)

    def test_valid_3_inf_rejected(self):
        raw = {h: 0.5 for h in HAZARDS}
        raw[HAZARDS[0]] = float("inf")
        with self.assertRaises(ValueError):
            validate_probs(raw)

    def test_valid_4_out_of_range_low(self):
        raw = {h: 0.5 for h in HAZARDS}
        raw[HAZARDS[0]] = -0.01
        with self.assertRaises(ValueError):
            validate_probs(raw)

    def test_valid_4_out_of_range_high(self):
        raw = {h: 0.5 for h in HAZARDS}
        raw[HAZARDS[0]] = 1.01
        with self.assertRaises(ValueError):
            validate_probs(raw)

    def test_valid_5_int_accepted_as_float(self):
        raw = {h: 0 for h in HAZARDS}
        result = validate_probs(raw)
        for v in result.values():
            self.assertIsInstance(v, float)

    def test_valid_5_boundary_values(self):
        raw = {h: 0.0 for h in HAZARDS}
        raw[HAZARDS[1]] = 1.0
        result = validate_probs(raw)
        self.assertEqual(result[HAZARDS[0]], 0.0)
        self.assertEqual(result[HAZARDS[1]], 1.0)

    def test_valid_not_dict(self):
        with self.assertRaises(ValueError) as ctx:
            validate_probs("not a dict")
        self.assertIn("invalid_output:", str(ctx.exception))


# ---------------------------------------------------------------------------
# EVAL-1..6: evaluate_tick
# ---------------------------------------------------------------------------

class TestEvaluateTick(unittest.TestCase):

    def setUp(self):
        self.cap, self.path = _make_cap(n_frames=15, fps=30.0)
        self.cfg = GateConfig()

    def tearDown(self):
        self.cap.release()
        os.unlink(self.path)

    # EVAL-1: success path fields
    def test_eval_1_success_fields(self):
        dec = _flat_decider(p=0.07)
        rec = evaluate_tick(dec, self.cap, tick_id=0, t_ms=3000, cfg=self.cfg)
        self.assertEqual(rec.status, "ok")
        self.assertEqual(rec.tick_id, 0)
        self.assertEqual(rec.t_ms, 3000)
        self.assertIsNotNone(rec.probs)
        self.assertAlmostEqual(rec.probs[HAZARDS[0]], 0.07, places=5)
        self.assertIsNone(rec.error)
        self.assertGreaterEqual(rec.jev_time_s, 0.0)
        self.assertIsInstance(rec.thumbs, list)
        # thumbs should contain 1..3 entries (indices {0, n//2, n-1})
        self.assertLessEqual(len(rec.thumbs), 3)
        for t in rec.thumbs:
            self.assertIsInstance(t, bytes)

    # EVAL-2: no_frames -> decider NOT called
    def test_eval_2_no_frames_decider_not_called(self):
        dec = _RecordingDecider()
        # force empty: t_ms=0 means window_start=0, end=0 -> 0ms window, linspace gives [0,0] but
        # a tiny video may still read at t=0. Use t_ms=1 and a high window -> window_start still 0
        # Just use a cap that has been released
        dead_cap, dead_path = _make_cap(n_frames=1, fps=1.0)
        dead_cap.release()
        # Attempt tick on a released cap — read will fail, no frames
        rec = evaluate_tick(dec, dead_cap, tick_id=99, t_ms=500, cfg=self.cfg)
        self.assertEqual(rec.status, "error")
        self.assertEqual(rec.error, "no_frames")
        self.assertEqual(dec.called, 0)   # decider NOT called
        self.assertIsNone(rec.probs)
        os.unlink(dead_path)

    # EVAL-3: decider exception -> "Type: msg"
    def test_eval_3_decider_exception(self):
        dec = _ErrorDecider()
        rec = evaluate_tick(dec, self.cap, tick_id=1, t_ms=3000, cfg=self.cfg)
        self.assertEqual(rec.status, "error")
        self.assertIn("ValueError", rec.error)
        self.assertIn("boom", rec.error)
        self.assertIsNone(rec.probs)

    # EVAL-4: invalid output -> "invalid_output:" prefix
    def test_eval_4_invalid_output(self):
        dec = _BadOutputDecider()
        rec = evaluate_tick(dec, self.cap, tick_id=2, t_ms=3000, cfg=self.cfg)
        self.assertEqual(rec.status, "error")
        self.assertTrue(rec.error.startswith("invalid_output:"))
        self.assertIsNone(rec.probs)

    # EVAL-5: jev_time_s >= delay_s
    def test_eval_5_jev_time_s_gte_delay(self):
        delay = 0.05
        dec = _flat_decider(delay_s=delay)
        rec = evaluate_tick(dec, self.cap, tick_id=3, t_ms=3000, cfg=self.cfg)
        self.assertGreaterEqual(rec.jev_time_s, delay)

    # EVAL-6: KeyboardInterrupt propagates
    def test_eval_6_keyboard_interrupt_propagates(self):
        dec = _KeyboardDecider()
        with self.assertRaises(KeyboardInterrupt):
            evaluate_tick(dec, self.cap, tick_id=4, t_ms=3000, cfg=self.cfg)


# ---------------------------------------------------------------------------
# MOCK-1..4: MockJevDecider
# ---------------------------------------------------------------------------

class TestMockJevDecider(unittest.TestCase):

    # MOCK-1: DEMO_RULES windows and defaults
    def test_mock_1_demo_rules_window(self):
        dec = MockJevDecider()
        # t_ms=6000 -> within DEMO_RULES (5500,8000)
        probs = dec([np.zeros((10, 10, 3), dtype=np.uint8)], 4.0, 6000)
        self.assertIn("no_ppe", probs)
        self.assertAlmostEqual(probs["no_ppe"], 0.25, places=5)
        # all other hazards get default 0.05
        for h in HAZARDS:
            if h != "no_ppe":
                self.assertAlmostEqual(probs[h], 0.05, places=5)

    # MOCK-1: t outside any rule -> all default
    def test_mock_1_outside_rules(self):
        dec = MockJevDecider()
        probs = dec([], 4.0, 1000)  # not in any rule
        for h in HAZARDS:
            self.assertAlmostEqual(probs[h], 0.05, places=5)

    # MOCK-1: second rule window (10000,12000)
    def test_mock_1_second_rule_window(self):
        dec = MockJevDecider()
        probs = dec([], 4.0, 11000)
        self.assertAlmostEqual(probs["no_ppe"], 0.60, places=5)

    # MOCK-2: first-rule-wins
    def test_mock_2_first_rule_wins(self):
        rules = [(0, 10000, {"no_ppe": 0.80}), (0, 10000, {"no_ppe": 0.10})]
        dec = MockJevDecider(rules=rules)
        probs = dec([], 4.0, 5000)
        self.assertAlmostEqual(probs["no_ppe"], 0.80, places=5)

    # MOCK-3: fail_at raises RuntimeError
    def test_mock_3_fail_at(self):
        dec = MockJevDecider(fail_at=(3000,))
        with self.assertRaises(RuntimeError) as ctx:
            dec([], 4.0, 3000)
        self.assertIn("3000", str(ctx.exception))

    # MOCK-3: bad_output_at drops blocked_exit
    def test_mock_3_bad_output_at(self):
        dec = MockJevDecider(bad_output_at=(3000,))
        result = dec([], 4.0, 3000)
        self.assertNotIn("blocked_exit", result)

    # MOCK-4: from_json round-trip
    def test_mock_4_from_json(self):
        import json
        data = {
            "default": 0.10,
            "delay_s": 0.0,
            "rules": [[5500, 8000, {"no_ppe": 0.40}]],
            "fail_at": [],
            "bad_output_at": [],
        }
        tmp = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
        json.dump(data, tmp)
        tmp.close()
        try:
            dec = MockJevDecider.from_json(tmp.name)
            probs = dec([], 4.0, 6000)
            self.assertAlmostEqual(probs["no_ppe"], 0.40, places=5)
            # unlisted hazards get default 0.10
            self.assertAlmostEqual(probs["blocked_exit"], 0.10, places=5)
        finally:
            os.unlink(tmp.name)

    # MOCK-4: unknown key -> ValueError
    def test_mock_4_unknown_key(self):
        import json
        data = {"default": 0.05, "unknown_field": 42}
        tmp = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
        json.dump(data, tmp)
        tmp.close()
        try:
            with self.assertRaises(ValueError):
                MockJevDecider.from_json(tmp.name)
        finally:
            os.unlink(tmp.name)

    # MOCK-4: empty rules list
    def test_mock_4_empty_rules(self):
        dec = MockJevDecider(rules=[])
        probs = dec([], 4.0, 5000)
        for h in HAZARDS:
            self.assertAlmostEqual(probs[h], 0.05, places=5)

    def test_hazard_questions_content(self):
        # HAZARD_QUESTIONS keys == HAZARDS and texts are non-empty strings
        self.assertEqual(set(HAZARD_QUESTIONS.keys()), set(HAZARDS))
        for k, v in HAZARD_QUESTIONS.items():
            self.assertIsInstance(v, str)
            self.assertGreater(len(v), 5)


# ---------------------------------------------------------------------------
# JEV-8: Lazy-import audit (I-5)
# ---------------------------------------------------------------------------

class TestLazyImport(unittest.TestCase):

    def test_jev_8_no_torch_or_transformers_without_onejev(self):
        """Lazy-import audit (I-5): runs in a fresh subprocess so that
        torch loaded by earlier tests cannot contaminate this check."""
        import subprocess
        import sys
        import os
        proj = r'C:\Users\Maha Mamdouh\Desktop\New folder (3)\New folder\UI_TEST\vlm_test_ui'
        script = (
            "import sys, os\n"
            f"sys.path.insert(0, {proj!r})\n"
            "import app.core.jev_adapter\n"
            "from examples.mock_jev import MockJevDecider as M\n"
            "dec = M()\n"
            "dec([], 4.0, 1000)\n"
            "assert 'torch' not in sys.modules, 'torch unexpectedly imported'\n"
            "assert 'transformers' not in sys.modules, 'transformers unexpectedly imported'\n"
            "print('OK')\n"
        )
        env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, text=True, timeout=30, env=env,
        )
        self.assertEqual(
            result.returncode, 0,
            f"Lazy-import violation:\n{result.stdout}\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()