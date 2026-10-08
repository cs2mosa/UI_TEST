"""
calibrate_jev.py - JEV Probability Calibration & Analysis Tool

Scans a video using OneJevDecider without running heavy HERMES generation.
Computes:
  1. Per-hazard score distributions (min, max, mean, p50, p90, p99).
  2. Peak risk timeline (timestamps with highest hazard scores).
  3. Grid simulation showing frame economy (% frames saved) across candidate thresholds.

Usage:
  python calibrate_jev.py --video "path/to/video.mp4" --jev-model-path "path/to/OneJev-0.8B/snapshot"
"""
from __future__ import annotations
import os
import sys
import argparse
import time
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "vlm_test_ui")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from app.core.jev_adapter import OneJevDecider, HAZARDS, evaluate_tick
from app.core.jev_gate import (
    GateConfig, GateRouter, compute_tick_times, assign_ticks_to_chunks,
    TIER_FULL, TIER_REDUCED,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate JEV Probability Thresholds on Video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video MP4")
    default_jev = os.environ.get("JEV_MODEL_PATH", os.path.join(BASE_DIR, ".hf_cache", "hub", "models--OmniJev--OneJev-0.8B", "snapshots", "c3939d8bf4cad34549a2b13bbb6aee9bcb6afee8"))
    parser.add_argument("--jev-model-path", type=str, default=default_jev, help="Path to OneJev-0.8B snapshot")
    parser.add_argument("--chunk-s", type=float, default=5.0, help="Chunk duration in seconds (default: 5.0)")
    parser.add_argument("--overlap-s", type=float, default=1.0, help="Chunk overlap in seconds (default: 1.0)")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional limit on number of chunks to scan")
    parser.add_argument("--output-csv", type=str, default="jev_calibration_log.csv", help="CSV export filename")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"[!] Video file not found: {args.video}")
        sys.exit(1)
    if not (os.path.isdir(args.jev_model_path) and any(os.scandir(args.jev_model_path))):
        print(f"[*] OneJev model path not found at: {args.jev_model_path}", flush=True)
        print("[*] Downloading 'OmniJev/OneJev-0.8B' via huggingface_hub...", flush=True)
        try:
            from huggingface_hub import snapshot_download
            hf_cache = os.environ.get("HF_HOME", os.path.join(BASE_DIR, ".hf_cache"))
            os.makedirs(hf_cache, exist_ok=True)
            args.jev_model_path = snapshot_download(repo_id="OmniJev/OneJev-0.8B", cache_dir=hf_cache)
            print(f"[+] OneJev ready at: {args.jev_model_path}", flush=True)
        except Exception as e:
            print(f"[!] Failed to auto-download OneJev: {e}")
            sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / fps if fps > 0 else 0
    duration_ms = int(duration_s * 1000)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 80)
    print("JEV PROBABILITY CALIBRATION RUNNER")
    print("=" * 80)
    print(f"Video File:     {args.video}")
    print(f"Resolution:     {w}x{h} @ {fps:.1f} fps")
    print(f"Duration:       {duration_s:.1f}s ({duration_s/60:.2f} min), Total Frames: {frame_count}")
    print(f"Model Path:     {args.jev_model_path}")

    # Build chunk schedule
    stride_ms = int((args.chunk_s - args.overlap_s) * 1000)
    chunk_dur_ms = int(args.chunk_s * 1000)
    windows = []
    start_ms = 0
    while start_ms < duration_ms:
        end_ms = min(duration_ms, start_ms + chunk_dur_ms)
        windows.append((start_ms, end_ms))
        if end_ms >= duration_ms:
            break
        start_ms += stride_ms

    if args.max_chunks:
        windows = windows[:args.max_chunks]
        duration_ms = windows[-1][1]

    gate_cfg = GateConfig()
    tick_times = compute_tick_times(duration_ms, gate_cfg.window_ms, gate_cfg.stride_ms)
    assigned = assign_ticks_to_chunks(tick_times, windows)

    print(f"Planned Chunks: {len(windows)}")
    print(f"Planned Ticks:  {len(tick_times)} (3s window, 1.5s stride)")
    print("-" * 80)

    print("[*] Initializing OneJevDecider on CUDA...")
    t0_load = time.perf_counter()
    decider = OneJevDecider(args.jev_model_path)
    print(f"[✓] OneJev loaded in {time.perf_counter() - t0_load:.2f}s")
    print("-" * 80)

    # Tick evaluation loop
    tick_results = []
    print(f"[*] Scanning {len(tick_times)} ticks across the video...")
    t0_scan = time.perf_counter()
    for i, t_ms in enumerate(tick_times):
        tr = evaluate_tick(decider, cap, i, t_ms, gate_cfg)
        tick_results.append(tr)
        if (i + 1) % 10 == 0 or (i + 1) == len(tick_times):
            pmax = max(tr.probs.values()) if tr.probs else 0.0
            elapsed = time.perf_counter() - t0_scan
            ms_per_tick = (elapsed / (i + 1)) * 1000.0
            print(f"  Tick {i+1}/{len(tick_times)} ({t_ms/1000:.1f}s) | pmax={pmax:.3f} | {ms_per_tick:.1f}ms/tick", flush=True)

    scan_time = time.perf_counter() - t0_scan
    cap.release()

    # 1. Per-Hazard Statistics
    print("\n" + "=" * 80)
    print(f"[1] PER-HAZARD PROBABILITY DISTRIBUTIONS ({len(tick_results)} ticks)")
    print("=" * 80)
    print(f"{'Hazard Name':<32} {'Min':<8} {'Max':<8} {'Mean':<8} {'p50':<8} {'p90':<8} {'p95':<8}")
    print("-" * 80)

    hazard_scores: dict[str, list[float]] = {h: [] for h in HAZARDS}
    for tr in tick_results:
        if tr.status == "ok" and tr.probs:
            for h in HAZARDS:
                hazard_scores[h].append(tr.probs.get(h, 0.0))

    for h in HAZARDS:
        arr = np.array(hazard_scores[h])
        if len(arr) > 0:
            print(f"{h:<32} {arr.min():<8.3f} {arr.max():<8.3f} {arr.mean():<8.3f} "
                  f"{np.percentile(arr, 50):<8.3f} {np.percentile(arr, 90):<8.3f} {np.percentile(arr, 95):<8.3f}")
        else:
            print(f"{h:<32} N/A")

    # 2. Risk Spike Timeline
    print("\n" + "=" * 80)
    print("[2] TOP 10 HIGHEST RISK MOMENTS (Spikes)")
    print("=" * 80)
    scored_ticks = []
    for tr in tick_results:
        if tr.status == "ok" and tr.probs:
            top_h = max(tr.probs.items(), key=lambda kv: kv[1])
            scored_ticks.append((tr.t_ms, top_h[0], top_h[1], tr.probs))
    scored_ticks.sort(key=lambda item: item[2], reverse=True)

    print(f"{'Time (s)':<12} {'Peak Hazard':<32} {'Score':<8} {'All Scores (abbreviated)'}")
    print("-" * 80)
    for t_ms, h_name, score, p_dict in scored_ticks[:10]:
        scores_str = ", ".join([f"{k[:4]}:{v:.2f}" for k, v in p_dict.items()])
        print(f"{t_ms/1000:<12.1f} {h_name:<32} {score:<8.3f} [{scores_str}]")

    # 3. Threshold Calibration Sweep Simulation
    print("\n" + "=" * 80)
    print("[3] THRESHOLD SWEEP SIMULATION (How thresholds affect routing & economy)")
    print("=" * 80)
    print("Testing candidate thresholds across the chunks...")
    candidate_thrs = [
        (0.70, 0.78),
        (0.75, 0.80),
        (0.78, 0.82),  # Calibrated Operating Point
        (0.80, 0.84),
        (0.82, 0.86),
    ]

    print(f"{'Watch Thr':<12} {'Full Thr':<12} {'Clear% (1fps)':<16} {'Full% (4fps)':<16} {'Frame Savings %'}")
    print("-" * 80)

    for watch_t, full_t in candidate_thrs:
        sim_cfg = GateConfig(
            route_watch_threshold=watch_t,
            route_full_threshold=full_t,
        )
        sim_router = GateRouter(sim_cfg)
        tier_counts = {TIER_FULL: 0, TIER_REDUCED: 0}
        total_sent = 0
        total_equiv = 0

        for idx, (w_start, w_end) in enumerate(windows):
            chunk_ticks = [tick_results[t_idx] for t_idx in assigned[idx]] if idx < len(assigned) else []
            dec = sim_router.decide(chunk_ticks)
            tier_counts[dec.tier] += 1

            equiv_frames = int(round(args.chunk_s * 4.0))  # 4 fps full
            sent_frames = int(round(args.chunk_s * (1.0 if dec.tier == TIER_REDUCED else 4.0)))
            total_sent += sent_frames
            total_equiv += equiv_frames

        tot_chunks = len(windows)
        clear_pct = (tier_counts[TIER_REDUCED] / tot_chunks) * 100.0 if tot_chunks else 0
        full_pct = (tier_counts[TIER_FULL] / tot_chunks) * 100.0 if tot_chunks else 0
        saved_pct = ((total_equiv - total_sent) / total_equiv) * 100.0 if total_equiv else 0
        curr_mark = " <- [CALIBRATED OPERATING POINT]" if (watch_t == 0.78 and full_t == 0.82) else ""
        print(f"{watch_t:<12.2f} {full_t:<12.2f} {clear_pct:<15.1f}% {full_pct:<15.1f}% {saved_pct:<15.1f}%{curr_mark}")

    # 4. Export CSV
    try:
        with open(args.output_csv, "w", encoding="utf-8") as f:
            f.write("tick_index,t_ms,t_sec,pmax," + ",".join(HAZARDS) + "\n")
            for i, tr in enumerate(tick_results):
                if tr.status == "ok" and tr.probs:
                    pmax = max(tr.probs.values())
                    vals = [f"{tr.probs.get(h, 0.0):.4f}" for h in HAZARDS]
                    f.write(f"{i},{tr.t_ms},{tr.t_ms/1000:.2f},{pmax:.4f}," + ",".join(vals) + "\n")
        print(f"\n[✓] Detailed per-tick probability log saved to: {args.output_csv}")
    except Exception as e:
        print(f"[!] Could not export CSV: {e}")

    print("\n" + "=" * 80)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
