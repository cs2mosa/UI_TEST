"""
g1_report.py - G1 analysis over a GatedBatchWorker gate log (design Appendix B).

Usage:
    python scripts/g1_report.py <gate_log.jsonl> [--demo-rules]

Pure Python, CPU-only, no Qt, no torch.
"""
import argparse
import json
import sys
from collections import Counter

ROUTE_FULL_THRESHOLD = 0.35
ROUTE_WATCH_THRESHOLD = 0.20


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING line {lineno}: {e}", file=sys.stderr)
    return rows


def partition(rows):
    headers, ticks, chunks = [], [], []
    for r in rows:
        k = r.get("kind")
        if k in ("run_start", "run_end"):
            headers.append(r)
        elif k == "tick":
            ticks.append(r)
        elif k == "chunk":
            chunks.append(r)
        else:
            headers.append(r)
    return headers, ticks, chunks


def report_header(headers, ticks, chunks):
    start = next((h for h in headers if h.get("kind") == "run_start"), {})
    end   = next((h for h in headers if h.get("kind") == "run_end"),   {})
    print("media:    ", start.get("media_path", "?"))
    print("decider:  ", start.get("decider_label", "?"))
    print("started:  ", start.get("ts", "?"))
    print("ended:    ", end.get("ts", "?"))
    print("ticks:    ", len(ticks))
    print("chunks:   ", len(chunks))


def check1_reasons(chunks):
    print("\n[1] Reasons histogram")
    if not chunks:
        print("  (no chunks)")
        return
    counts = Counter(c.get("reason", "?") for c in chunks)
    total = len(chunks)
    for reason, n in counts.most_common():
        print(f"  {reason}: {n} ({n/total*100:.0f}%)")
    print(f"  total: {total}")


def check2_frame_economy(chunks):
    print("\n[2] Frame economy")
    if not chunks:
        print("  (no chunks)")
        return
    sent  = sum(c.get("frames_sent", 0) for c in chunks)
    equiv = sum(c.get("frames_full_equiv", 0) for c in chunks)
    print(f"  frames_sent:       {sent}")
    print(f"  frames_full_equiv: {equiv}")
    if equiv > 0:
        print(f"  frames_saved_pct:  {(equiv - sent) / equiv * 100:.1f}%")
    else:
        print("  frames_saved_pct:  n/a (equiv=0)")

    full = [c["vlm_time_s"] for c in chunks if c.get("tier") == "FULL"  and c.get("vlm_time_s") is not None]
    red  = [c["vlm_time_s"] for c in chunks if c.get("tier") == "REDUCED" and c.get("vlm_time_s") is not None]
    if full:
        print(f"  avg vlm_time FULL:    {sum(full)/len(full):.2f}s (n={len(full)})")
    if red:
        print(f"  avg vlm_time REDUCED: {sum(red)/len(red):.2f}s (n={len(red)})")


def check3_overhead(chunks):
    print("\n[3] Overhead ratio")
    if not chunks:
        print("  (no chunks)")
        return
    jev   = sum(c.get("jev_time_s", 0.0) for c in chunks)
    vlm   = sum(c["vlm_time_s"] for c in chunks if c.get("vlm_time_s") is not None)
    sent  = sum(c.get("frames_sent", 0) for c in chunks)
    equiv = sum(c.get("frames_full_equiv", 0) for c in chunks)
    ratio = jev / vlm if vlm > 0 else float("inf")
    print(f"  sum jev_time_s: {jev:.2f}s")
    print(f"  sum vlm_time_s: {vlm:.2f}s")
    print(f"  overhead ratio (jev/vlm): {ratio:.3f}")
    saved_frames = equiv - sent
    print(f"  frames saved: {saved_frames}")
    if vlm > 0 and equiv > 0:
        est_saved = vlm * saved_frames / equiv
        print(f"  est vlm time saved: {est_saved:.2f}s")
        print(f"  net saving (est_saved - jev): {est_saved - jev:.2f}s")


def check4_window_correctness(ticks):
    print("\n[4] Window correctness (DEMO_RULES pseudo-GT)")
    if not ticks:
        print("  (no tick records)")
        return
    full_total, full_ok   = 0, 0
    watch_total, watch_ok = 0, 0
    fails = []
    for t in ticks:
        t_ms  = t.get("t_ms", -1)
        no_ppe = (t.get("vec") or {}).get("no_ppe", 0.0)
        if 10000 <= t_ms <= 12000:
            full_total += 1
            if no_ppe >= ROUTE_FULL_THRESHOLD:
                full_ok += 1
            else:
                fails.append(f"  FAIL t={t_ms}ms no_ppe={no_ppe:.3f} < {ROUTE_FULL_THRESHOLD} (full window)")
        if 5500 <= t_ms <= 8000:
            watch_total += 1
            if no_ppe >= ROUTE_WATCH_THRESHOLD:
                watch_ok += 1
            else:
                fails.append(f"  FAIL t={t_ms}ms no_ppe={no_ppe:.3f} < {ROUTE_WATCH_THRESHOLD} (watch window)")
    print(f"  full-window  [10000-12000ms]: {full_ok}/{full_total} ticks have no_ppe >= {ROUTE_FULL_THRESHOLD}")
    print(f"  watch-window [ 5500- 8000ms]: {watch_ok}/{watch_total} ticks have no_ppe >= {ROUTE_WATCH_THRESHOLD}")
    if fails:
        for f in fails:
            print(f)
    else:
        print("  OK - no violations")


def check5_no_downrating(chunks):
    print("\n[5] No violation chunk down-rated")
    bad = [
        c for c in chunks
        if c.get("tier") == "REDUCED"
        and not (c.get("end_ms", 0) < 10000 or c.get("start_ms", 0) > 12000)
    ]
    if bad:
        print(f"  FAIL - {len(bad)} REDUCED chunk(s) overlap [10000-12000ms]:")
        for c in bad:
            print(f"    chunk_id={c.get('chunk_id')} [{c.get('start_ms')}-{c.get('end_ms')}] reason={c.get('reason')}")
    else:
        print("  OK - 0 violation chunks down-rated")


def check6_early_warning(chunks):
    print("\n[6] Early warning")
    above_full   = [c for c in chunks if c.get("reason") == "above_full"]
    early_signal = [c for c in chunks if c.get("reason") in ("above_watch", "rising")]
    first_full  = min((c.get("start_ms", 0) for c in above_full),   default=None)
    first_early = min((c.get("start_ms", 0) for c in early_signal), default=None)
    print(f"  first above_full start_ms:          {first_full}")
    print(f"  first above_watch/rising start_ms:  {first_early}")
    if first_full is None:
        print("  WARNING - no above_full chunk found")
    elif first_early is None:
        print("  FAIL - no early signal before first above_full")
    elif first_early < first_full:
        print("  OK - early signal precedes first above_full")
    else:
        print(f"  FAIL - early signal ({first_early}ms) does not precede above_full ({first_full}ms)")


def main():
    parser = argparse.ArgumentParser(description="G1 analysis over a GatLog JSONL (Appendix B).")
    parser.add_argument("gate_log", help="Path to .jsonl gate log file")
    parser.add_argument("--demo-rules", action="store_true",
                        help="Enable window-correctness check using DEMO_RULES as pseudo-GT")
    args = parser.parse_args()

    rows = load(args.gate_log)
    headers, ticks, chunks = partition(rows)

    print(f"G1 report: {args.gate_log}")
    report_header(headers, ticks, chunks)

    check1_reasons(chunks)
    check2_frame_economy(chunks)
    check3_overhead(chunks)
    if args.demo_rules:
        check4_window_correctness(ticks)
    else:
        print("\n[4] Window correctness - skipped (pass --demo-rules to enable)")
    check5_no_downrating(chunks)
    check6_early_warning(chunks)
    print("\nDone.")


if __name__ == "__main__":
    main()
