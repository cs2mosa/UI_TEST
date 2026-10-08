from __future__ import annotations
"""
test_calibrate_report.py - Phase C CAL-1..6 tests (design_v3 §10.1, Appendix D).

Pinned operative values on the toy set (single hazard no_ppe; item-level pmax:
pos [0.9, 0.8, 0.3, 0.2], neg [0.7, 0.6, 0.4, 0.1]): AUC 0.625; candidates are the
seven midpoints (plus min-eps / max+eps); t=0.65 -> precision 2/3 recall 0.5;
t=0.75 -> precision 1.0 recall 0.5; floor 0.60 selects t=0.65.

CPU-only; stdlib; no app imports; no GPU; no network. calibrate_report.py is loaded
from scripts/ via importlib (it is not a package module).
"""
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest

_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "calibrate_report.py"))
_spec = importlib.util.spec_from_file_location("calibrate_report", _SCRIPT)
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)


def _toy_items():
    """Appendix D toy: 8 items, one row each, single hazard no_ppe."""
    pos = [0.9, 0.8, 0.3, 0.2]
    neg = [0.7, 0.6, 0.4, 0.1]
    items = []
    for i, p in enumerate(pos):
        items.append({"item_id": f"pos_{i}", "rows": [{"hazard": "no_ppe", "p_yes": p, "label": "pos"}]})
    for i, p in enumerate(neg):
        items.append({"item_id": f"neg_{i}", "rows": [{"hazard": "no_ppe", "p_yes": p, "label": "neg"}]})
    return items


def _write_toy_jsonl(path: str, pos_scores, neg_scores, hazard="no_ppe"):
    meta = {"type": "run_meta", "prompt_hash": "deadbeef", "seed": 42}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for i, p in enumerate(pos_scores):
            f.write(json.dumps({"source": "isafety", "item_id": f"pos_{i}", "hazard": hazard,
                                "p_yes": p, "label": "pos", "tags": [], "order": "ab",
                                "camera": None, "meta": {"duration_ms": 5000, "n_frames": 20}}) + "\n")
        for i, p in enumerate(neg_scores):
            f.write(json.dumps({"source": "isafety", "item_id": f"neg_{i}", "hazard": hazard,
                                "p_yes": p, "label": "neg", "tags": [], "order": "ab",
                                "camera": None, "meta": {"duration_ms": 5000, "n_frames": 20}}) + "\n")


class TestCal1Auc(unittest.TestCase):
    def test_appendix_d_auc_with_ties(self):
        pos = [0.9, 0.8, 0.3, 0.2]
        neg = [0.7, 0.6, 0.4, 0.1]
        self.assertAlmostEqual(cr.auc(pos, neg), 0.625, places=9)

    def test_ties_count_half(self):
        self.assertAlmostEqual(cr.auc([0.5], [0.5]), 0.5, places=9)
        self.assertAlmostEqual(cr.auc([1.0], [0.0]), 1.0, places=9)
        self.assertIsNone(cr.auc([], [0.5]))


class TestCal2Candidates(unittest.TestCase):
    def test_midpoint_candidates(self):
        pmaxes = [0.9, 0.8, 0.3, 0.2, 0.7, 0.6, 0.4, 0.1]
        cands = cr.candidate_thresholds(pmaxes)
        self.assertAlmostEqual(cands[0], 0.1 - cr.CANDIDATE_EPS, places=12)
        self.assertAlmostEqual(cands[-1], 0.9 + cr.CANDIDATE_EPS, places=12)
        mids = cands[1:-1]
        for got, want in zip(mids, [0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(len(mids), 7)


class TestCal3PrecisionFloorSelection(unittest.TestCase):
    def test_appendix_d_points(self):
        items = _toy_items()
        m065 = cr.routing_counts(0.65, items)
        self.assertAlmostEqual(m065["precision"], 2.0 / 3.0, places=9)
        self.assertAlmostEqual(m065["recall"], 0.5, places=9)
        m075 = cr.routing_counts(0.75, items)
        self.assertAlmostEqual(m075["precision"], 1.0, places=9)
        self.assertAlmostEqual(m075["recall"], 0.5, places=9)

    def test_floor_060_selects_065(self):
        sel = cr.select_operating_point(_toy_items(), precision_floor=0.60, watch_floor=0.40)
        self.assertAlmostEqual(sel["full_t"], 0.65, places=12)
        # watch: smallest candidate with precision >= 0.40 and <= full (the min-eps
        # candidate routes everything -> toy base rate 0.5 >= 0.40)
        self.assertLessEqual(sel["watch_t"], sel["full_t"])

    def test_floor_unreachable(self):
        # top value belongs to a negative: no candidate exceeds precision 2/3
        items = [{"item_id": f"pos_{i}", "rows": [{"hazard": "no_ppe", "p_yes": p, "label": "pos"}]}
                 for i, p in enumerate([0.9, 0.8])]
        items += [{"item_id": f"neg_{i}", "rows": [{"hazard": "no_ppe", "p_yes": p, "label": "neg"}]}
                  for i, p in enumerate([0.95, 0.1])]
        sel = cr.select_operating_point(items, precision_floor=0.95, watch_floor=0.40)
        self.assertIsNone(sel["full_t"])


class TestCal4SplitDeterminism(unittest.TestCase):
    def _items_n(self, n_pos, n_neg):
        items = []
        for i in range(n_pos):
            items.append({"item_id": f"p{i:02d}", "rows": [{"hazard": "no_ppe", "p_yes": 0.9, "label": "pos"}]})
        for i in range(n_neg):
            items.append({"item_id": f"n{i:02d}", "rows": [{"hazard": "no_ppe", "p_yes": 0.1, "label": "neg"}]})
        return items

    def test_same_seed_identical_split(self):
        items = self._items_n(4, 4)
        t1, h1 = cr.split_items(items, seed=42)
        t2, h2 = cr.split_items(items, seed=42)
        self.assertEqual([x["item_id"] for x in t1], [x["item_id"] for x in t2])
        self.assertEqual([x["item_id"] for x in h1], [x["item_id"] for x in h2])

    def test_odd_stratum_extra_item_to_tune(self):
        items = self._items_n(5, 4)  # hazardous stratum odd (5) -> 3 tune / 2 holdout
        tune, holdout = cr.split_items(items, seed=7)
        tune_pos = [x for x in tune if x["item_id"].startswith("p")]
        hold_pos = [x for x in holdout if x["item_id"].startswith("p")]
        self.assertEqual(len(tune_pos), 3)
        self.assertEqual(len(hold_pos), 2)
        self.assertEqual(len(tune) + len(holdout), 9)
        ids_t = {x["item_id"] for x in tune}
        ids_h = {x["item_id"] for x in holdout}
        self.assertFalse(ids_t & ids_h)

    def test_rows_stay_with_item(self):
        items = [{"item_id": "multi", "rows": [
            {"hazard": h, "p_yes": 0.5, "label": "neg"} for h in
            ("pedestrian_vehicle", "no_ppe", "blocked_exit", "unsafe_load", "fall_or_spill")]},
            {"item_id": "solo", "rows": [{"hazard": "no_ppe", "p_yes": 0.9, "label": "pos"}]}]
        tune, holdout = cr.split_items(items, seed=3)
        all_ids = [x["item_id"] for x in tune + holdout]
        self.assertEqual(sorted(all_ids), ["multi", "solo"])
        for x in tune + holdout:
            self.assertEqual(len(x["rows"]), 5 if x["item_id"] == "multi" else 1)


class TestCal5EndToEndAdoption(unittest.TestCase):
    def test_adopted_writes_thresholds_with_legacy_rows(self):
        # homogeneous strata (8 pos @ 0.95, 2 neg @ 0.1): every split keeps the
        # holdout base rate at 0.8, so the adoption gates pass for any seed
        with tempfile.TemporaryDirectory() as d:
            scores = os.path.join(d, "scores.jsonl")
            out = os.path.join(d, "thresholds.json")
            _write_toy_jsonl(scores, pos_scores=[0.95] * 8, neg_scores=[0.1, 0.1])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cr.main([scores, "--precision-floor", "0.6", "--watch-precision-floor", "0.4",
                              "--seed", "42", "--out", out])
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            for key in ("route_watch_threshold", "route_full_threshold", "precision_floor",
                        "watch_precision_floor", "seed", "prompt_hash", "dataset_fingerprint",
                        "selection", "holdout", "null_input", "order_bias", "created"):
                self.assertIn(key, doc)
            for name in ("legacy_020_035", "legacy_078_082"):
                self.assertIn(name, doc["holdout"])
                for metric in ("precision", "recall", "f1"):
                    self.assertIn(metric, doc["holdout"][name])
            self.assertLessEqual(doc["route_watch_threshold"], doc["route_full_threshold"])

    def test_report_only_recomputes(self):
        with tempfile.TemporaryDirectory() as d:
            scores = os.path.join(d, "scores.jsonl")
            out = os.path.join(d, "thresholds.json")
            _write_toy_jsonl(scores, pos_scores=[0.95] * 8, neg_scores=[0.1, 0.1])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cr.main([scores, "--out", out]), 0)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cr.main([scores, "--report-only", out])
            self.assertEqual(rc, 0)
            self.assertIn("[holdout report]", buf.getvalue())


class TestCal6GatesAndBootstrap(unittest.TestCase):
    def test_bootstrap_reproducible(self):
        items = _toy_items()
        _tune, holdout = cr.split_items(items, seed=42)
        b1 = cr.bootstrap(holdout, watch_t=0.65, seed=42)
        b2 = cr.bootstrap(holdout, watch_t=0.65, seed=42)
        self.assertEqual(b1, b2)

    def test_exit_code_3_no_thresholds_file(self):
        # interleaved scores: no candidate reaches precision 0.99 -> FAILED-TO-ADOPT
        with tempfile.TemporaryDirectory() as d:
            scores = os.path.join(d, "scores.jsonl")
            out = os.path.join(d, "thresholds.json")
            _write_toy_jsonl(scores, pos_scores=[0.6, 0.5, 0.55, 0.45], neg_scores=[0.55, 0.45, 0.5, 0.4])
            err = io.StringIO()
            out_buf = io.StringIO()
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err):
                rc = cr.main([scores, "--precision-floor", "0.99", "--out", out])
            self.assertEqual(rc, 3)
            self.assertFalse(os.path.exists(out))
            self.assertIn("FAILED-TO-ADOPT", err.getvalue())


class TestGateConfigPlumbing(unittest.TestCase):
    """3WAY-2 (design_v3 §10.1): --gate-config-json plumbing with a fake config file;
    the worker builds the config from it; explicit flags override the JSON."""

    def _load_cl(self):
        spec = importlib.util.spec_from_file_location(
            "compare_latency_for_plumbing",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "compare_latency.py")),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _fake_config(self, d):
        import json as _json
        path = os.path.join(d, "fake_thresholds.json")
        with open(path, "w", encoding="utf-8") as f:
            _json.dump({"route_watch_threshold": 0.3, "route_full_threshold": 0.5}, f)
        return path

    def test_worker_builds_config_from_json(self):
        from app.core.jev_gate import GateConfig
        cl = self._load_cl()
        with tempfile.TemporaryDirectory() as d:
            path = self._fake_config(d)
            cfg = cl.build_gate_config(path, None, None)
            self.assertIsInstance(cfg, GateConfig)
            self.assertAlmostEqual(cfg.route_watch_threshold, 0.3, places=9)
            self.assertAlmostEqual(cfg.route_full_threshold, 0.5, places=9)

    def test_explicit_flags_override_json(self):
        cl = self._load_cl()
        with tempfile.TemporaryDirectory() as d:
            path = self._fake_config(d)
            cfg = cl.build_gate_config(path, 0.7, 0.8)
            self.assertAlmostEqual(cfg.route_watch_threshold, 0.7, places=9)
            self.assertAlmostEqual(cfg.route_full_threshold, 0.8, places=9)

    def test_missing_file_raises(self):
        cl = self._load_cl()
        with self.assertRaises(FileNotFoundError):
            cl.build_gate_config("does_not_exist.json", None, None)

    def test_invalid_payload_raises(self):
        cl = self._load_cl()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"route_full_threshold": 5.0, "route_watch_threshold": 0.2}')
            with self.assertRaises(ValueError):
                cl.build_gate_config(path, None, None)


if __name__ == "__main__":
    unittest.main()
