from __future__ import annotations
"""
calibrate_report.py - JEV threshold selection on labeled OneJev scores (design_v3 §5.4).

CPU-only; stdlib only; no app imports. Deterministic; seeded. Input: labeled_scores.jsonl
(Appendix C: first line run_meta, then one row per (item, hazard)). The split is by ITEM,
never by row. Selection is recall-first subject to precision floors (D-25); hard adoption
gates: holdout recall >= 0.95 AND precision >= the full floor (§5.4 step 6); legacy points
(0.20/0.35 and 0.78/0.82, D-30) are reported, never adopted without the gates.

Exit codes: 0 adopted (thresholds.json written); 3 gates failed / floor unreachable
(no thresholds.json); 2 schema or IO error.

Appendix D note: on the pinned toy set the operative values are AUC 0.625, the seven
midpoint candidates, t=0.65 -> precision 2/3 recall 0.5, t=0.75 -> precision 1.0 recall
0.5, and floor 0.60 selecting t=0.65. (The document's inline precision labels for
t=0.15/0.25 are swapped relative to arithmetic: t=0.15 -> 4/7, t=0.25 -> 0.5. Reported,
not edited - the selection is unaffected.)
"""
import argparse
import json
import math
import random
import sys
import time

HAZARDS = ("pedestrian_vehicle", "no_ppe", "blocked_exit", "unsafe_load", "fall_or_spill")
LEGACY_POINTS = [("legacy_020_035", 0.20, 0.35), ("legacy_078_082", 0.78, 0.82)]
CANDIDATE_EPS = 1e-6
RECALL_GATE = 0.95
WEAK_AUC = 0.65
ECE_BINS = 15
RELIABILITY_BINS = 5
BOOTSTRAP_N = 1000


# ---------------------------------------------------------------------------
# Loading + validation (Appendix C schema; abort with the line number, exit 2)
# ---------------------------------------------------------------------------

def load_rows(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as exc:
        raise SchemaError(f"cannot read {path}: {exc}")
    if not lines:
        raise SchemaError(f"{path}: empty file")
    try:
        run_meta = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise SchemaError(f"{path}: line 1 is not JSON: {exc}")
    if run_meta.get("type") != "run_meta":
        raise SchemaError(f"{path}: line 1 must be a run_meta record")
    items: dict[str, dict] = {}
    null_rows: list[dict] = []
    probe_rows: list[dict] = []
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{path}: line {lineno} is not JSON: {exc}")
        _validate_row(row, path, lineno)
        if row["source"] == "null_input":
            null_rows.append(row)
        elif row["source"] == "order_probe":
            probe_rows.append(row)
        else:
            item = items.setdefault(row["item_id"], {"item_id": row["item_id"], "rows": [], "tags": row["tags"], "camera": row["camera"]})
            item["rows"].append(row)
    return run_meta, items, null_rows, probe_rows


def _validate_row(row, path: str, lineno: int) -> None:
    required = {"source", "item_id", "hazard", "p_yes", "label", "tags", "order", "camera", "meta"}
    missing = required - set(row)
    if missing:
        raise SchemaError(f"{path}: line {lineno} missing keys {sorted(missing)}")
    if row["source"] not in ("isafety", "null_input", "order_probe"):
        raise SchemaError(f"{path}: line {lineno} bad source {row['source']!r}")
    if row["hazard"] not in HAZARDS:
        raise SchemaError(f"{path}: line {lineno} bad hazard {row['hazard']!r}")
    p = row["p_yes"]
    if not isinstance(p, (int, float)) or isinstance(p, bool) or not (0.0 <= float(p) <= 1.0):
        raise SchemaError(f"{path}: line {lineno} p_yes out of range: {p!r}")
    if row["label"] not in ("pos", "neg", "null"):
        raise SchemaError(f"{path}: line {lineno} bad label {row['label']!r}")
    if row["order"] not in ("ab", "ba"):
        raise SchemaError(f"{path}: line {lineno} bad order {row['order']!r}")


class SchemaError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Metrics (AUC ties 0.5; ECE and Brier copied from qev.calibrate definitions)
# ---------------------------------------------------------------------------

def auc(pos_scores: list[float], neg_scores: list[float]) -> float | None:
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos_scores) * len(neg_scores))


def ece(confidences: list[float], correct: list[float], n_bins: int = ECE_BINS) -> float:
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c <= lo)]
        if idx:
            avg_conf = sum(confidences[i] for i in idx) / len(idx)
            avg_acc = sum(correct[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(avg_conf - avg_acc)
    return total


def brier(probs: list[float], target: list[float]) -> float:
    return sum((p - t) ** 2 for p, t in zip(probs, target))


def reliability_table(confidences: list[float], correct: list[float], n_bins: int = RELIABILITY_BINS) -> list[dict]:
    table = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c <= lo)]
        table.append({
            "bin": f"({lo:.1f},{hi:.1f}]",
            "n": len(idx),
            "mean_conf": (sum(confidences[i] for i in idx) / len(idx)) if idx else None,
            "mean_acc": (sum(correct[i] for i in idx) / len(idx)) if idx else None,
        })
    return table


# ---------------------------------------------------------------------------
# Item-level helpers (split is by item, never by row - §5.4 step 2)
# ---------------------------------------------------------------------------

def item_pmax(item: dict) -> float | None:
    scores = [float(r["p_yes"]) for r in item["rows"]]
    return max(scores) if scores else None


def item_hazardous(item: dict) -> bool:
    return any(r["label"] == "pos" for r in item["rows"])


def split_items(items: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified by hazardous; within a stratum sort by item_id, shuffle with
    random.Random(seed); first half of the items -> tune, second -> holdout
    (odd counts: the extra item goes to tune). Every row of an item stays with it."""
    rng = random.Random(seed)
    tune: list[dict] = []
    holdout: list[dict] = []
    for stratum in (True, False):
        members = sorted((it for it in items if item_hazardous(it) == stratum), key=lambda x: x["item_id"])
        rng.shuffle(members)
        half = (len(members) + 1) // 2
        tune.extend(members[:half])
        holdout.extend(members[half:])
    return tune, holdout


# ---------------------------------------------------------------------------
# Routing scan (§5.4 steps 4-5)
# ---------------------------------------------------------------------------

def candidate_thresholds(pmaxes: list[float]) -> list[float]:
    vals = sorted(set(pmaxes))
    if not vals:
        return []
    cands = [vals[0] - CANDIDATE_EPS]
    for a, b in zip(vals, vals[1:]):
        cands.append((a + b) / 2.0)
    cands.append(vals[-1] + CANDIDATE_EPS)
    return cands


def routing_counts(threshold: float, items: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for it in items:
        routed = item_pmax(it) >= threshold
        haz = item_hazardous(it)
        if routed and haz:
            tp += 1
        elif routed and not haz:
            fp += 1
        elif not routed and haz:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}


def scan_table(items: list[dict]) -> list[dict]:
    pmaxes = [p for p in (item_pmax(it) for it in items) if p is not None]
    return [
        {"t": t, **routing_counts(t, items)}
        for t in candidate_thresholds(pmaxes)
    ]


def select_operating_point(items: list[dict], precision_floor: float, watch_floor: float) -> dict:
    table = scan_table(items)
    full_t = None
    for row in table:
        if row["precision"] >= precision_floor:
            full_t = row["t"]
            break
    if full_t is None:
        return {"table": table, "full_t": None, "watch_t": None, "degenerate_watch": False}
    watch_t = None
    for row in table:
        if row["precision"] >= watch_floor and row["t"] <= full_t:
            watch_t = row["t"]
            break
    degenerate = watch_t is None
    if degenerate:
        watch_t = full_t
    return {"table": table, "full_t": full_t, "watch_t": watch_t, "degenerate_watch": degenerate}


def f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def quantile(values: list[float], q: float) -> float:
    """Computes empirical quantile q in [0, 1] for values."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(math.floor(round(q * len(s), 9)))))
    return s[idx]


def calibrate_neyman_pearson(
    items: list[dict],
    target_recall: float = 0.97,
    default_full: float = 0.35,
    default_watch: float = 0.20,
) -> dict:
    """
    Computes per-hazard full and watch thresholds using the Neyman-Pearson criterion:
    For hazard h, threshold is set to the (1.0 - target_recall) quantile of positive scores.
    Watch threshold is set to the (1.0 - target_recall)/2 quantile (higher recall).
    If a hazard has no positive instances (e.g. no_ppe in iSafetyBench), falls back to defaults.
    """
    hazard_scores: dict[str, list[float]] = {h: [] for h in HAZARDS}
    for it in items:
        for r in it["rows"]:
            if r["hazard"] in HAZARDS and r["label"] == "pos":
                hazard_scores[r["hazard"]].append(float(r["p_yes"]))

    full_thrs: dict[str, float] = {}
    watch_thrs: dict[str, float] = {}
    stats: dict[str, dict] = {}

    alpha = max(0.0, 1.0 - target_recall)
    for h in HAZARDS:
        pos = hazard_scores[h]
        if pos:
            # Neyman-Pearson: alpha-quantile of positive scores
            t_full = max(0.01, quantile(pos, alpha))
            t_watch = max(0.005, min(t_full, quantile(pos, alpha / 2.0)))
            if t_watch >= t_full:
                t_watch = max(0.005, round(t_full * 0.75, 4))
            stats[h] = {
                "n_pos": len(pos),
                "min": min(pos),
                "max": max(pos),
                "mean": sum(pos) / len(pos),
                "status": "calibrated",
            }
        else:
            t_full = default_full
            t_watch = default_watch
            stats[h] = {
                "n_pos": 0,
                "status": "defaulted_no_data",
            }
        full_thrs[h] = round(t_full, 4)
        watch_thrs[h] = round(t_watch, 4)

    return {
        "hazard_full_thresholds": full_thrs,
        "hazard_watch_thresholds": watch_thrs,
        "hazard_stats": stats,
        "overall_full_threshold": min(full_thrs.values()),
        "overall_watch_threshold": min(watch_thrs.values()),
        "target_recall": target_recall,
    }


def per_hazard_routing_counts(hazard_thrs: dict[str, float], items: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for it in items:
        routed = any(
            float(r["p_yes"]) >= hazard_thrs.get(r["hazard"], 1.0)
            for r in it["rows"]
            if r["hazard"] in hazard_thrs
        )
        haz = item_hazardous(it)
        if routed and haz:
            tp += 1
        elif routed and not haz:
            fp += 1
        elif not routed and haz:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# Diagnostics (§5.4 step 3)
# ---------------------------------------------------------------------------

def hazard_diagnostics(items: list[dict], hazard: str) -> dict:
    pos, neg = [], []
    for it in items:
        for r in it["rows"]:
            if r["hazard"] != hazard or r["label"] == "null":
                continue
            (pos if r["label"] == "pos" else neg).append(float(r["p_yes"]))
    conf = pos + neg
    correct = [1.0] * len(pos) + [0.0] * len(neg)
    return {
        "n_pos": len(pos),
        "n_neg": len(neg),
        "auc": auc(pos, neg),
        "brier": brier(conf, correct) if conf else None,
        "ece": ece(conf, correct) if conf else None,
        "reliability": reliability_table(conf, correct) if conf else [],
    }


def diagnostics(items: list[dict]) -> dict:
    out = {}
    for h in HAZARDS:
        out[h] = hazard_diagnostics(items, h)
        if out[h]["auc"] is not None and out[h]["auc"] < WEAK_AUC:
            out[h]["weak_ranker"] = True
    return out


# ---------------------------------------------------------------------------
# Holdout evaluation (§5.4 step 6) + bootstrap (step 7)
# ---------------------------------------------------------------------------

def point_metrics(watch_t: float, items: list[dict]) -> dict:
    counts = routing_counts(watch_t, items)
    return {
        "precision": counts["precision"],
        "recall": counts["recall"],
        "f1": f1(counts["precision"], counts["recall"]),
        "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"], "tn": counts["tn"],
    }


def per_hazard_holdout(threshold_or_dict: float | dict[str, float], items: list[dict]) -> dict:
    out = {}
    for h in HAZARDS:
        thr = threshold_or_dict[h] if isinstance(threshold_or_dict, dict) else threshold_or_dict
        sub = []
        for it in items:
            scores = [float(r["p_yes"]) for r in it["rows"] if r["hazard"] == h]
            if not scores:
                continue
            sub.append({"p": max(scores), "pos": any(r["hazard"] == h and r["label"] == "pos" for r in it["rows"])})
        tp = sum(1 for x in sub if x["p"] >= thr and x["pos"])
        fp = sum(1 for x in sub if x["p"] >= thr and not x["pos"])
        fn = sum(1 for x in sub if x["p"] < thr and x["pos"])
        out[h] = {
            "threshold": round(thr, 4),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 1.0,
            "n_pos": sum(1 for x in sub if x["pos"]),
        }
    return out


def bootstrap(holdout: list[dict], watch_t: float, seed: int) -> dict:
    rng = random.Random(seed)
    n = len(holdout)
    recalls, precisions, aucs = [], [], []
    for _ in range(BOOTSTRAP_N):
        sample = [holdout[rng.randrange(n)] for _ in range(n)]
        m = point_metrics(watch_t, sample)
        recalls.append(m["recall"])
        precisions.append(m["precision"])
        pos = [float(r["p_yes"]) for it in sample for r in it["rows"] if r["label"] == "pos"]
        neg = [float(r["p_yes"]) for it in sample for r in it["rows"] if r["label"] == "neg"]
        a = auc(pos, neg)
        if a is not None:
            aucs.append(a)

    def ci(values: list[float]) -> list[float]:
        if not values:
            return [None, None]
        s = sorted(values)
        lo = s[max(0, int(0.025 * len(s)) - 1)]
        hi = s[min(len(s) - 1, int(0.975 * len(s)))]
        return [lo, hi]

    return {
        "recall_ci95": ci(recalls),
        "precision_ci95": ci(precisions),
        "pooled_auc_ci95": ci(aucs),
    }


# ---------------------------------------------------------------------------
# Null-input and order-bias tables (§5.4 steps 8-9; report only)
# ---------------------------------------------------------------------------

def null_input_table(null_rows: list[dict]) -> dict:
    groups: dict[str, dict[str, list[float]]] = {}
    for r in null_rows:
        cam = r["camera"] or "global"
        groups.setdefault(cam, {}).setdefault(r["hazard"], []).append(float(r["p_yes"]))
    out = {}
    for cam, per_hazard in groups.items():
        entry = {}
        for h, vals in per_hazard.items():
            mean = sum(vals) / len(vals)
            entry[h] = {
                "mean_p_yes": mean,
                "logit_offset": math.log(mean / (1.0 - mean)) if 0.0 < mean < 1.0 else None,
            }
        out[cam] = entry
    return out


def order_bias_table(probe_rows: list[dict]) -> dict:
    pairs: dict[tuple, dict[str, float]] = {}
    for r in probe_rows:
        key = (r["item_id"], r["hazard"])
        pairs.setdefault(key, {})[r["order"]] = float(r["p_yes"])
    acc: dict[str, list[float]] = {h: [] for h in HAZARDS}
    for (item_id, h), vals in pairs.items():
        if "ab" in vals and "ba" in vals:
            acc[h].append(abs(vals["ab"] - vals["ba"]))
    return {h: (sum(v) / len(v) if v else None) for h, v in acc.items()}


# ---------------------------------------------------------------------------
# Printing + outputs
# ---------------------------------------------------------------------------

def print_diagnostics(title: str, diags: dict) -> None:
    print(f"\n[{title} diagnostics]")
    for h, d in diags.items():
        if d["auc"] is None:
            print(f"  {h}: no labeled rows")
            continue
        weak = " [weak_ranker]" if d.get("weak_ranker") else ""
        print(f"  {h}: n_pos={d['n_pos']} n_neg={d['n_neg']} auc={d['auc']:.3f} "
              f"brier={d['brier']:.4f} ece={d['ece']:.4f}{weak}")


def print_scan(title: str, table: list[dict]) -> None:
    print(f"\n[{title} routing scan] t | routed TP/FP | precision | recall")
    for r in table:
        print(f"  t={r['t']:.6f} TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']} "
              f"precision={r['precision']:.3f} recall={r['recall']:.3f}")


def print_holdout(selected: dict, holdout_metrics: dict) -> None:
    print("\n[holdout report] point | precision | recall | f1")
    for name, m in holdout_metrics.items():
        print(f"  {name}: precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")
    print(f"  selected per-hazard breakdown (full threshold):")
    for h, m in selected["per_hazard"].items():
        print(f"    {h}: precision={m['precision']:.3f} recall={m['recall']:.3f} (n_pos={m['n_pos']})")


def build_thresholds(run_meta, args, selection, tune, holdout_items, holdout_metrics, null_rows, probe_rows,
                     precision_floor: float = 0.0, watch_precision_floor: float = 0.0) -> dict:
    watch_t, full_t = selection["watch_t"], selection["full_t"]
    per_haz_pos = {h: 0 for h in HAZARDS}
    per_haz_neg = {h: 0 for h in HAZARDS}
    for it in tune + holdout_items:
        for r in it["rows"]:
            if r["label"] == "pos":
                per_haz_pos[r["hazard"]] += 1
            elif r["label"] == "neg":
                per_haz_neg[r["hazard"]] += 1
    return {
        "route_watch_threshold": watch_t,
        "route_full_threshold": full_t,
        "hazard_full_thresholds": selection.get("hazard_full_thresholds"),
        "hazard_watch_thresholds": selection.get("hazard_watch_thresholds"),
        "calibration_method": selection.get("method", getattr(args, "mode", "neyman_pearson")),
        "precision_floor": precision_floor,
        "watch_precision_floor": watch_precision_floor,
        "target_recall": getattr(args, "target_recall", None),
        "seed": args.seed,
        "prompt_hash": run_meta.get("prompt_hash", ""),
        "dataset_fingerprint": {
            "n_items": len(tune) + len(holdout_items),
            "n_tune": len(tune),
            "n_holdout": len(holdout_items),
            "n_pos": sum(per_haz_pos.values()),
            "n_neg": sum(per_haz_neg.values()),
            "per_hazard": {h: {"pos": per_haz_pos[h], "neg": per_haz_neg[h]} for h in HAZARDS},
        },
        "selection": {
            "tune_stats": {
                "candidates": selection["table"],
                "degenerate_watch": selection["degenerate_watch"],
                "hazard_stats": selection.get("hazard_stats", {}),
            },
        },
        "holdout": holdout_metrics,
        "null_input": null_input_table(null_rows),
        "order_bias": order_bias_table(probe_rows),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="JEV threshold selection report (design_v3 §5.4)")
    parser.add_argument("labeled_scores", help="labeled_scores.jsonl (Appendix C)")
    parser.add_argument("--mode", choices=["neyman_pearson", "grid_scan"], default="neyman_pearson",
                        help="calibration algorithm (default: neyman_pearson)")
    parser.add_argument("--target-recall", type=float, default=0.97,
                        help="target recall for Neyman-Pearson calibration (default: 0.97)")
    parser.add_argument("--target-recall-gate", type=float, default=0.95,
                        help="minimum holdout recall adoption gate (default: 0.95)")
    parser.add_argument("--default-full", type=float, default=0.35,
                        help="fallback full threshold for unmapped hazards (default: 0.35)")
    parser.add_argument("--default-watch", type=float, default=0.20,
                        help="fallback watch threshold for unmapped hazards (default: 0.20)")
    parser.add_argument("--precision-floor", type=float, default=None,
                        help="minimum precision floor (default: 0.60 for grid_scan, 0.0 for neyman_pearson)")
    parser.add_argument("--watch-precision-floor", type=float, default=None,
                        help="minimum watch precision floor (default: 0.40 for grid_scan, 0.0 for neyman_pearson)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="thresholds.json")
    parser.add_argument("--report-only", default=None, metavar="THRESHOLDS_JSON",
                        help="recompute holdout metrics for an existing thresholds.json")
    args = parser.parse_args(argv)

    precision_floor = args.precision_floor if args.precision_floor is not None else (0.6 if args.mode == "grid_scan" else 0.0)
    watch_precision_floor = args.watch_precision_floor if args.watch_precision_floor is not None else (0.4 if args.mode == "grid_scan" else 0.0)

    try:
        run_meta, items_map, null_rows, probe_rows = load_rows(args.labeled_scores)
    except SchemaError as exc:
        print(f"SCHEMA ERROR: {exc}", file=sys.stderr)
        return 2

    items = list(items_map.values())
    for it in items:
        it["pmax"] = item_pmax(it)
        it["hazardous"] = item_hazardous(it)
    print(f"[*] {len(items)} items "
          f"({sum(1 for i in items if i['hazardous'])} hazardous, "
          f"{sum(1 for i in items if not i['hazardous'])} clean-negative); "
          f"{len(null_rows)} null rows; {len(probe_rows)} order-probe rows")
    if not items:
        print("FAILED-TO-ADOPT (no items)", file=sys.stderr)
        return 3

    tune, holdout = split_items(items, args.seed)
    print(f"[*] split (seed={args.seed}): tune={len(tune)} holdout={len(holdout)}")

    print_diagnostics("tune", diagnostics(tune))
    print_diagnostics("holdout", diagnostics(holdout))

    if args.report_only is not None:
        try:
            with open(args.report_only, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"SCHEMA ERROR: cannot read {args.report_only}: {exc}", file=sys.stderr)
            return 2
        watch_t = saved.get("route_watch_threshold", 0.20)
        full_t = saved.get("route_full_threshold", 0.35)
        saved_haz_full = saved.get("hazard_full_thresholds")
        saved_haz_watch = saved.get("hazard_watch_thresholds")

        target_full = saved_haz_full if saved_haz_full else full_t

        selected = {
            "watch_t": watch_t,
            "full_t": full_t,
            "hazard_full_thresholds": saved_haz_full,
            "hazard_watch_thresholds": saved_haz_watch,
            "per_hazard": per_hazard_holdout(target_full, holdout),
        }
        if saved_haz_watch:
            sel_point_m = per_hazard_routing_counts(saved_haz_watch, holdout)
            sel_entry = {
                "precision": sel_point_m["precision"],
                "recall": sel_point_m["recall"],
                "f1": f1(sel_point_m["precision"], sel_point_m["recall"]),
                "tp": sel_point_m["tp"], "fp": sel_point_m["fp"],
                "fn": sel_point_m["fn"], "tn": sel_point_m["tn"],
            }
        else:
            sel_entry = point_metrics(watch_t, holdout)

        holdout_metrics = {"selected": sel_entry}
        for n, lw, lf in LEGACY_POINTS:
            holdout_metrics[n] = point_metrics(lw, holdout)

        print_holdout(selected, holdout_metrics)
        m = holdout_metrics["selected"]
        gates_ok = m["recall"] >= args.target_recall_gate and m["precision"] >= precision_floor
        print(f"[gates] recall {m['recall']:.3f} >= {args.target_recall_gate} and precision {m['precision']:.3f} "
              f">= {precision_floor}: {'PASS' if gates_ok else 'FAIL'}")
        return 0 if gates_ok else 3

    if args.mode == "neyman_pearson":
        np_res = calibrate_neyman_pearson(
            tune,
            target_recall=args.target_recall,
            default_full=args.default_full,
            default_watch=args.default_watch,
        )
        full_thrs = np_res["hazard_full_thresholds"]
        watch_thrs = np_res["hazard_watch_thresholds"]
        full_t = np_res["overall_full_threshold"]
        watch_t = np_res["overall_watch_threshold"]

        selection = {
            "method": "neyman_pearson",
            "table": scan_table(tune),
            "full_t": full_t,
            "watch_t": watch_t,
            "hazard_full_thresholds": full_thrs,
            "hazard_watch_thresholds": watch_thrs,
            "hazard_stats": np_res["hazard_stats"],
            "degenerate_watch": False,
        }
        print(f"\n[neyman-pearson per-hazard calibration] target_recall={args.target_recall:.3f}")
        for h in HAZARDS:
            st = np_res["hazard_stats"][h]
            print(f"  {h:<24} full={full_thrs[h]:.4f} watch={watch_thrs[h]:.4f} ({st['status']})")
        print(f"[*] overall fallback bounds: watch={watch_t:.4f} full={full_t:.4f}")

        sel_counts = per_hazard_routing_counts(watch_thrs, holdout)
        sel_m = {
            "precision": sel_counts["precision"],
            "recall": sel_counts["recall"],
            "f1": f1(sel_counts["precision"], sel_counts["recall"]),
            "tp": sel_counts["tp"], "fp": sel_counts["fp"],
            "fn": sel_counts["fn"], "tn": sel_counts["tn"],
        }
        holdout_metrics = {"selected": sel_m}
        for n, lw, fw in LEGACY_POINTS:
            holdout_metrics[n] = point_metrics(lw, holdout)
        selected = {
            "watch_t": watch_t,
            "full_t": full_t,
            "hazard_full_thresholds": full_thrs,
            "hazard_watch_thresholds": watch_thrs,
            "per_hazard": per_hazard_holdout(full_thrs, holdout),
        }
    else:
        selection = select_operating_point(tune, precision_floor, watch_precision_floor)
        print_scan("tune", selection["table"])
        if selection["full_t"] is None:
            print("FAILED-TO-ADOPT (precision floor unreachable)", file=sys.stderr)
            return 3
        if selection["degenerate_watch"]:
            print("[flag] degenerate_watch: no candidate meets the watch floor below the full threshold; watch := full")
        print(f"[*] selected operating point: watch={selection['watch_t']:.6f} full={selection['full_t']:.6f}")

        holdout_metrics = {name: point_metrics(w, holdout) for name, _f, w in
                           [("selected", 0.0, selection["watch_t"])] + [(n, fw, lw) for n, lw, fw in LEGACY_POINTS]}
        selected = {"watch_t": selection["watch_t"], "full_t": selection["full_t"],
                    "per_hazard": per_hazard_holdout(selection["full_t"], holdout)}

    print_holdout(selected, holdout_metrics)

    sel_m = holdout_metrics["selected"]
    gates_ok = sel_m["recall"] >= args.target_recall_gate and sel_m["precision"] >= precision_floor
    print(f"[gates] recall {sel_m['recall']:.3f} >= {args.target_recall_gate} and precision "
          f"{sel_m['precision']:.3f} >= {precision_floor}: {'PASS' if gates_ok else 'FAIL'}")

    for name, lw, fw in LEGACY_POINTS:
        lm = holdout_metrics[name]
        if (lm["recall"] >= args.target_recall_gate and lm["precision"] >= precision_floor
                and lm["f1"] > sel_m["f1"]):
            print(f"ADVISORY: legacy point dominates selected ({name} f1={lm['f1']:.3f} > selected f1={sel_m['f1']:.3f}) - developer decides")

    print(f"[bootstrap] {json.dumps(bootstrap(holdout, selection['watch_t'], args.seed))}")
    null_tab = null_input_table(null_rows)
    if null_tab:
        print(f"[null-input] {json.dumps(null_tab, sort_keys=True)}")
    bias = order_bias_table(probe_rows)
    if any(v is not None for v in bias.values()):
        print(f"[order-bias] {json.dumps(bias)}")

    if not gates_ok:
        print("FAILED-TO-ADOPT (adoption gates failed); thresholds.json NOT written", file=sys.stderr)
        return 3

    doc = build_thresholds(run_meta, args, selection, tune, holdout, holdout_metrics, null_rows, probe_rows,
                           precision_floor=precision_floor, watch_precision_floor=watch_precision_floor)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"[+] thresholds written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
