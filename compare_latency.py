"""
Automated Latency & Performance Benchmark: HERMES vs. InfiniPot-V

This script conducts an isolated, rigorous, and honest head-to-head comparison
between the HERMES streaming VLM engine (URCA_PROJECTS/vlm_test_ui) and the
InfiniPot-V streaming VLM engine (temper/vlm_test_ui).

To eliminate external confounders:
1. Subprocess Isolation: Each model runs in its own clean Python process so
   the 6 GB VRAM GPU is 100% dedicated to one model at a time.
2. Identical Frames & Chunks: Frames are decoded and sampled identically
   before being fed into both runners.
3. GPU Timing Synchronization: Every stage uses torch.cuda.synchronize()
   to prevent asynchronous CUDA queue distortion.
4. Stage Breakdown: Ingestion, Compression, and Generation are tracked per chunk.
"""

from __future__ import annotations
import os
import sys
import argparse
import json
import time
import subprocess
from typing import List, Dict, Any
import numpy as np
import cv2
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_default_candidate = os.path.join(
    BASE_DIR,
    ".hf_cache", "hub",
    "models--Qwen--Qwen3-VL-4B-Instruct",
    "snapshots", "ebb281ec70b05090aa6165b016eac8ec08e71b17",
)
if not os.path.exists(_default_candidate):
    _default_candidate = (
        r"F:\URCA_PROJECTS\.hf_cache\hub"
        r"\models--Qwen--Qwen3-VL-4B-Instruct"
        r"\snapshots\ebb281ec70b05090aa6165b016eac8ec08e71b17"
    )
MODEL_PATH = os.environ.get("MODEL_PATH", _default_candidate)

_default_jev_candidate = os.path.join(
    BASE_DIR,
    ".hf_cache", "hub",
    "models--OmniJev--OneJev-0.8B",
    "snapshots", "c3939d8bf4cad34549a2b13bbb6aee9bcb6afee8",
)
if not os.path.exists(_default_jev_candidate):
    _default_jev_candidate = (
        r"F:\URCA_PROJECTS\.hf_cache\hub"
        r"\models--OmniJev--OneJev-0.8B"
        r"\snapshots\c3939d8bf4cad34549a2b13bbb6aee9bcb6afee8"
    )
JEV_MODEL_PATH = os.environ.get("JEV_MODEL_PATH", _default_jev_candidate)

DEFAULT_PROMPT = (
    "The video shows an industrial or surveillance scene. One or more people or "
    "vehicles may be present. Review the video carefully. Describe what is "
    "happening in one or two sentences, focusing on any safety hazard or "
    "violation. Then rate the severity of the safety incident from 0.0 to 1.0:\n"
    "0.0-0.2 routine activity, no hazard; 0.3-0.5 unsafe behaviour without "
    "immediate danger; 0.6-0.8 clear safety violation or near-miss; "
    "0.9-1.0 accident or serious danger in progress.\n"
    'Reply with JSON only: {"caption": "<text>", "score": <number from 0 to 1>}'
)


def ensure_model_path(model_path: str, repo_id: str = "Qwen/Qwen3-VL-4B-Instruct") -> str:
    """Ensures model snapshot exists locally; downloads it automatically if missing."""
    if os.path.isdir(model_path) and any(os.scandir(model_path)):
        return model_path

    print(f"[*] Local model directory not found at: {model_path}", flush=True)
    print(f"[*] Downloading '{repo_id}' via huggingface_hub...", flush=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[!] huggingface_hub not installed, cannot download model automatically.", flush=True)
        return model_path

    hf_cache = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not hf_cache:
        hf_cache = os.path.join(BASE_DIR, ".hf_cache")

    os.makedirs(hf_cache, exist_ok=True)
    downloaded_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=hf_cache,
    )
    print(f"[+] Model ready at: {downloaded_dir}", flush=True)
    return downloaded_dir


def sample_video_chunks(
    video_path: str,
    num_chunks: int = 5,
    chunk_s: float = 5.0,
    overlap_s: float = 1.0,
    vlm_fps: float = 4.0,
    max_dim: int = 448,
) -> List[Dict[str, Any]]:
    """Extract identical video chunks in memory for both runners."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_ms = int(round((frame_count / source_fps) * 1000.0))

    step_s = chunk_s - overlap_s
    chunk_ms = int(chunk_s * 1000.0)
    step_ms = int(step_s * 1000.0)

    windows = []
    start_ms = 0
    while start_ms < duration_ms and len(windows) < num_chunks:
        end_ms = min(start_ms + chunk_ms, duration_ms)
        windows.append((start_ms, end_ms))
        if end_ms >= duration_ms:
            break
        start_ms += step_ms

    chunks_data = []
    for chunk_id, (w_start, w_end) in enumerate(windows):
        duration_sec = (w_end - w_start) / 1000.0
        target_frame_count = max(2, int(round(duration_sec * vlm_fps)))
        sample_ts_list = np.linspace(w_start, w_end, target_frame_count, dtype=int).tolist()

        frames = []
        timestamps = []
        for ts in sample_ts_list:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts)
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if max_dim is not None and max_dim > 0 and max(rgb.shape[0], rgb.shape[1]) > max_dim:
                scale = max_dim / max(rgb.shape[0], rgb.shape[1])
                new_w, new_h = max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
            timestamps.append(int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        if not frames:
            cap.set(cv2.CAP_PROP_POS_MSEC, w_start)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if max_dim is not None and max_dim > 0 and max(rgb.shape[0], rgb.shape[1]) > max_dim:
                    scale = max_dim / max(rgb.shape[0], rgb.shape[1])
                    new_w, new_h = max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))
                    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frames.append(rgb)
                timestamps.append(w_start)

        chunks_data.append({
            "chunk_id": chunk_id,
            "start_ms": w_start,
            "end_ms": w_end,
            "fps": vlm_fps,
            "frames": frames,
            "timestamps_ms": timestamps,
        })

    cap.release()
    return chunks_data


# ---------------------------------------------------------------------------
# Worker: HERMES
# ---------------------------------------------------------------------------
def run_hermes_worker(args):
    # Set sys.path strictly to vlm_test_ui (evict any temper paths)
    hermes_root = os.path.abspath(os.path.join(BASE_DIR, "vlm_test_ui"))
    infinipot_root = os.path.abspath(os.path.join(BASE_DIR, "temper", "vlm_test_ui"))
    sys.path = [p for p in sys.path if os.path.abspath(p) != infinipot_root and os.path.abspath(p) != hermes_root]
    sys.path.insert(0, hermes_root)
    for mod in list(sys.modules.keys()):
        if mod == "vlm" or mod.startswith("vlm.") or mod == "app" or mod.startswith("app."):
            del sys.modules[mod]

    from vlm.hermes_runner import HermesStreamingRunner
    from app.core.types import VideoChunk

    print(f"[*] Initializing HERMES worker from {args.model_path}...", flush=True)
    runner = HermesStreamingRunner(
        model_path=args.model_path,
        prompt=args.prompt,
        kv_size=args.kv_size,
        sample_fps=0.5,
        load_in_4bit=True,
        max_new_tokens=args.max_new_tokens,
        encode_chunk_size=8,
    )

    chunks_data = sample_video_chunks(
        video_path=args.video,
        num_chunks=args.num_chunks,
        chunk_s=args.chunk_s,
        overlap_s=args.overlap_s,
        vlm_fps=args.vlm_fps,
        max_dim=args.max_dim,
    )

    runner.start_session()
    results = []

    # Reset max memory tracker
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for idx, cdata in enumerate(chunks_data):
        chunk = VideoChunk(
            chunk_id=cdata["chunk_id"],
            source=args.video,
            start_ms=cdata["start_ms"],
            end_ms=cdata["end_ms"],
            fps=cdata["fps"],
            frames=cdata["frames"],
            timestamps_ms=cdata["timestamps_ms"],
        )
        num_frames = len(chunk.frames)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Step 1: Chunk Ingestion (Video Encoding & Prefill)
        frames_array = np.stack(chunk.frames, axis=0)
        video_tensor = torch.from_numpy(frames_array)
        total_frames = video_tensor.shape[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_ingest_start = time.perf_counter()

        def _kv_len():
            return runner.model.kv_cache[0][0].shape[2] if runner.model.kv_cache is not None else 0

        kv_pre_ingest = _kv_len()
        for start in range(0, total_frames, runner.encode_chunk_size):
            sub = video_tensor[start : start + runner.encode_chunk_size]
            runner.model.encode_video_chunk(sub, frame_fps=cdata["fps"])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_ingest = time.perf_counter() - t_ingest_start
        kv_pre_compress = _kv_len()
        vision_tokens = kv_pre_compress - kv_pre_ingest

        # Step 2: Safety-Aware KV Compression
        t_compress_start = time.perf_counter()
        runner.model.predict_and_compress()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_compress = time.perf_counter() - t_compress_start
        kv_post_compress = _kv_len()

        # Step 3: Generation (Question Answering)
        t_gen_start = time.perf_counter()
        input_text = {
            "question": runner.prompt,
            "prompt": runner.model.get_prompt(runner.prompt),
        }
        raw_output = runner.model.question_answering(
            input_text, max_new_tokens=runner.max_new_tokens
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gen = time.perf_counter() - t_gen_start

        t_total = time.perf_counter() - t0
        latency_per_frame_ms = (t_total / num_frames) * 1000.0

        # Memory introspection
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        # KV Cache length inspection
        kv_len = runner.model.kv_cache[0][0].shape[2] if runner.model.kv_cache else 0

        parsed = runner._parse(raw_output)
        print(f"[HERMES] Chunk {idx+1}/{len(chunks_data)}: Total={t_total:.3f}s | Ingest={t_ingest:.3f}s | "
              f"Compress={t_compress:.3f}s | Gen={t_gen:.3f}s | Latency/frame={latency_per_frame_ms:.1f}ms | "
              f"VisTok={vision_tokens} | Grid={runner.model.last_video_grid_thw} | KV={kv_len}", flush=True)

        results.append({
            "chunk_id": idx,
            "num_frames": num_frames,
            "total_time_s": t_total,
            "ingest_time_s": t_ingest,
            "compress_time_s": t_compress,
            "gen_time_s": t_gen,
            "latency_per_frame_ms": latency_per_frame_ms,
            "vision_tokens_ingested": vision_tokens,
            "kv_len_pre_compress": kv_pre_compress,
            "kv_len_post_compress": kv_post_compress,
            "video_grid_thw": runner.model.last_video_grid_thw,
            "kv_cache_len": kv_len,
            "peak_vram_mb": peak_vram_mb,
            "caption": parsed.caption,
            "score": parsed.score,
        })

    runner.end_session()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"engine": "HERMES", "chunks": results}, f, indent=2)
    print(f"[+] HERMES worker completed. Results saved to {args.output}", flush=True)


# ---------------------------------------------------------------------------
# Worker: JEV+HERMES  (gated; S.11.2)
# ---------------------------------------------------------------------------
def run_jev_hermes_worker(args):
    """Gated benchmark worker: OneJev decides tier, then HERMES runs on the routed frames."""
    # path isolation: same as run_hermes_worker
    hermes_root = os.path.abspath(os.path.join(BASE_DIR, "vlm_test_ui"))
    infinipot_root = os.path.abspath(os.path.join(BASE_DIR, "temper", "vlm_test_ui"))
    sys.path = [p for p in sys.path if os.path.abspath(p) != infinipot_root and os.path.abspath(p) != hermes_root]
    sys.path.insert(0, hermes_root)
    for mod in list(sys.modules.keys()):
        if mod == "vlm" or mod.startswith("vlm.") or mod == "app" or mod.startswith("app."):
            del sys.modules[mod]

    # lazy-import JEV (I-5: torch/transformers loaded only in the worker subprocess)
    import torch
    if not torch.cuda.is_available():
        print("[!] JEV+HERMES worker requires CUDA -- no CUDA device found.", flush=True)
        sys.exit(1)
    if not os.path.isdir(args.jev_model_path):
        print(f"[!] OneJev model not found: {args.jev_model_path}", flush=True)
        sys.exit(1)

    from vlm.hermes_runner import HermesStreamingRunner
    from app.core.types import VideoChunk
    from app.core.jev_gate import (
        GateConfig, GateRouter,
        compute_tick_times, assign_ticks_to_chunks,
        TIER_FULL, TIER_REDUCED,
    )
    from app.core.jev_adapter import OneJevDecider, evaluate_tick
    from app.core.frame_sampler import (
        sample_timestamps, read_frames_at, resize_max_side,
    )
    import cv2 as _cv2

    gate_cfg = build_gate_config(args.gate_config_json, args.watch_threshold, args.full_threshold)

    print(f"[*] Loading OneJev from {args.jev_model_path}...", flush=True)
    decider = OneJevDecider(args.jev_model_path)

    print(f"[*] Initializing HERMES runner from {args.model_path}...", flush=True)
    runner = HermesStreamingRunner(
        model_path=args.model_path,
        prompt=args.prompt,
        kv_size=args.kv_size,
        sample_fps=0.5,
        load_in_4bit=True,
        max_new_tokens=args.max_new_tokens,
        encode_chunk_size=8,
    )

    chunks_data = sample_video_chunks(
        video_path=args.video,
        num_chunks=args.num_chunks,
        chunk_s=args.chunk_s,
        overlap_s=args.overlap_s,
        vlm_fps=args.vlm_fps,
        max_dim=args.max_dim,
    )

    # Compute JEV tick schedule once (S.7.1 / S.11.2)
    cap_duration_ms = int(chunks_data[-1]["end_ms"]) if chunks_data else 0
    tick_times = compute_tick_times(cap_duration_ms, gate_cfg.window_ms, gate_cfg.stride_ms)
    windows    = [(c["start_ms"], c["end_ms"]) for c in chunks_data]
    assigned   = assign_ticks_to_chunks(tick_times, windows)

    router = GateRouter(gate_cfg)
    runner.start_session()
    results = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cap_jev = _cv2.VideoCapture(args.video)

    for idx, cdata in enumerate(chunks_data):
        start_ms, end_ms = cdata["start_ms"], cdata["end_ms"]
        chunk_tick_times = assigned[idx] if idx < len(assigned) else []

        # JEV phase
        t_jev_start = time.perf_counter()
        tick_records = []
        for t_ms in chunk_tick_times:
            tr = evaluate_tick(decider, cap_jev, len(tick_records), t_ms, gate_cfg)
            tick_records.append(tr)
        jev_time_s = time.perf_counter() - t_jev_start

        decision = router.decide(tick_records)
        tier   = decision.tier
        reason = decision.reason
        pmax   = decision.pmax

        # Frame selection
        if tier == TIER_REDUCED:
            ts_list = sample_timestamps(start_ms, end_ms, gate_cfg.route_reduced_fps)
            frames_rgb, _ = read_frames_at(cap_jev, ts_list, rgb=True)
            frames_to_send = [resize_max_side(f, args.max_dim) for f in frames_rgb] if frames_rgb else cdata["frames"]
        else:
            frames_to_send = cdata["frames"]  # FULL: bit-identical to hermes engine

        frames_sent       = len(frames_to_send)
        frames_full_equiv = len(cdata["frames"])

        # HERMES phase (identical instrumentation to run_hermes_worker)
        chunk = VideoChunk(
            chunk_id=cdata["chunk_id"],
            source=args.video,
            start_ms=start_ms,
            end_ms=end_ms,
            fps=cdata["fps"],
            frames=frames_to_send,
            timestamps_ms=cdata["timestamps_ms"][:frames_sent],
        )
        num_frames = len(chunk.frames)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Step 1: Chunk Ingestion (Video Encoding & Prefill)
        frames_array = np.stack(chunk.frames, axis=0)
        video_tensor = torch.from_numpy(frames_array)
        total_frames = video_tensor.shape[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_ingest_start = time.perf_counter()

        def _kv_len():
            return runner.model.kv_cache[0][0].shape[2] if runner.model.kv_cache is not None else 0

        kv_pre_ingest = _kv_len()
        for start in range(0, total_frames, runner.encode_chunk_size):
            sub = video_tensor[start : start + runner.encode_chunk_size]
            runner.model.encode_video_chunk(sub, frame_fps=chunk.fps)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_ingest = time.perf_counter() - t_ingest_start
        kv_pre_compress = _kv_len()
        vision_tokens = kv_pre_compress - kv_pre_ingest

        # Step 2: Safety-Aware KV Compression
        t_compress_start = time.perf_counter()
        runner.model.predict_and_compress()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_compress = time.perf_counter() - t_compress_start
        kv_post_compress = _kv_len()

        # Step 3: Generation (Question Answering)
        t_gen_start = time.perf_counter()
        input_text = {
            "question": runner.prompt,
            "prompt": runner.model.get_prompt(runner.prompt),
        }
        raw_output = runner.model.question_answering(
            input_text, max_new_tokens=runner.max_new_tokens
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gen = time.perf_counter() - t_gen_start

        t_hermes         = time.perf_counter() - t0
        t_total          = t_hermes + jev_time_s   # gross: JEV + HERMES (S.11.3)
        net_total_time_s = t_hermes                # net: HERMES only
        latency_per_frame_ms = (t_total / num_frames) * 1000.0
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        kv_len = runner.model.kv_cache[0][0].shape[2] if runner.model.kv_cache else 0

        parsed = runner._parse(raw_output)
        print(
            f"[JEV+HERMES] Chunk {idx+1}/{len(chunks_data)}: tier={tier} reason={reason} "
            f"frames={frames_sent}/{frames_full_equiv} JEV={jev_time_s:.3f}s "
            f"Total={t_total:.3f}s Net={net_total_time_s:.3f}s",
            flush=True,
        )

        results.append({
            # all existing keys (S.11.2)
            "chunk_id": idx,
            "num_frames": num_frames,
            "total_time_s": t_total,
            "ingest_time_s": t_ingest,
            "compress_time_s": t_compress,
            "gen_time_s": t_gen,
            "latency_per_frame_ms": latency_per_frame_ms,
            "vision_tokens_ingested": vision_tokens,
            "kv_len_pre_compress": kv_pre_compress,
            "kv_len_post_compress": kv_post_compress,
            "kv_cache_len": kv_len,
            "peak_vram_mb": peak_vram_mb,
            "caption": parsed.caption,
            "score": parsed.score,
            # JEV-specific extras
            "jev_time_s": jev_time_s,
            "jev_tier": tier,
            "jev_reason": reason,
            "jev_pmax": pmax,
            "frames_sent": frames_sent,
            "frames_full_equiv": frames_full_equiv,
            "net_total_time_s": net_total_time_s,
        })

    cap_jev.release()
    runner.end_session()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"engine": "JEV+HERMES", "chunks": results}, f, indent=2)
    print(f"[+] JEV+HERMES worker completed. Results saved to {args.output}", flush=True)


# ---------------------------------------------------------------------------
# Worker: InfiniPot-V
# ---------------------------------------------------------------------------
def run_infinipot_worker(args):
    # Set sys.path strictly to temper/vlm_test_ui (evict any hermes paths)
    infinipot_root = os.path.abspath(os.path.join(BASE_DIR, "temper", "vlm_test_ui"))
    hermes_root = os.path.abspath(os.path.join(BASE_DIR, "vlm_test_ui"))
    sys.path = [p for p in sys.path if os.path.abspath(p) != hermes_root and os.path.abspath(p) != infinipot_root]
    sys.path.insert(0, infinipot_root)
    for mod in list(sys.modules.keys()):
        if mod == "vlm" or mod.startswith("vlm.") or mod == "app" or mod.startswith("app."):
            del sys.modules[mod]

    from vlm.qwen3vl_runner import Qwen3VLRunner
    from app.core.types import VideoChunk

    print(f"[*] Initializing InfiniPot-V worker from {args.model_path}...", flush=True)
    runner = Qwen3VLRunner(
        model_path=args.model_path,
        prompt=args.prompt,
        num_frames=16,
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=True,
        memory_budget_frames=16,
        compress_to_frames=8,
        compression_method="infinipot-v",
        use_infinipot=True,
    )

    chunks_data = sample_video_chunks(
        video_path=args.video,
        num_chunks=args.num_chunks,
        chunk_s=args.chunk_s,
        overlap_s=args.overlap_s,
        vlm_fps=args.vlm_fps,
        max_dim=args.max_dim,
    )

    runner.reset()
    results = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for idx, cdata in enumerate(chunks_data):
        chunk = VideoChunk(
            chunk_id=cdata["chunk_id"],
            source=args.video,
            start_ms=cdata["start_ms"],
            end_ms=cdata["end_ms"],
            fps=cdata["fps"],
            frames=cdata["frames"],
            timestamps_ms=cdata["timestamps_ms"],
        )
        num_frames = len(chunk.frames)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        def _kv_len():
            return runner.past_key_values.layers[0].keys.shape[2] if runner.past_key_values is not None else 0

        kv_pre_ingest = _kv_len()

        # In Qwen3VLRunner, _append_chunk_to_cache runs vision encoding,
        # appends to cache, and immediately triggers _maybe_compress.
        # We hook _maybe_compress to measure exact compression time and
        # capture KV length before/after compression.
        orig_maybe_compress = runner._maybe_compress
        compress_duration = [0.0]
        compress_bounds = [kv_pre_ingest, kv_pre_ingest]  # kv len before/after compression

        def timed_maybe_compress(budget, target):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            tc0 = time.perf_counter()
            compress_bounds[0] = _kv_len()
            orig_maybe_compress(budget, target)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            compress_duration[0] = time.perf_counter() - tc0
            compress_bounds[1] = _kv_len()

        runner._maybe_compress = timed_maybe_compress

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_ingest_start = time.perf_counter()
        runner._append_chunk_to_cache(chunk)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_append_total = time.perf_counter() - t_ingest_start

        # Restore original method
        runner._maybe_compress = orig_maybe_compress

        t_compress = compress_duration[0]
        t_ingest = max(0.0, t_append_total - t_compress)
        vision_tokens = max(0, compress_bounds[0] - kv_pre_ingest)

        # Step 2: Generation
        t_gen_start = time.perf_counter()
        raw_output = runner._generate_from_cache(runner.prompt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gen = time.perf_counter() - t_gen_start

        t_total = time.perf_counter() - t0
        latency_per_frame_ms = (t_total / num_frames) * 1000.0

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        # Current KV Cache seq length
        kv_len = runner.past_key_values.layers[0].keys.shape[2] if runner.past_key_values else 0

        parsed = runner._parse(raw_output)
        print(f"[InfiniPot-V] Chunk {idx+1}/{len(chunks_data)}: Total={t_total:.3f}s | Ingest={t_ingest:.3f}s | "
              f"Compress={t_compress:.3f}s | Gen={t_gen:.3f}s | Latency/frame={latency_per_frame_ms:.1f}ms | "
              f"VisTok={vision_tokens} | Grid={runner.last_video_grid_thw} | KV={kv_len}", flush=True)

        results.append({
            "chunk_id": idx,
            "num_frames": num_frames,
            "total_time_s": t_total,
            "ingest_time_s": t_ingest,
            "compress_time_s": t_compress,
            "gen_time_s": t_gen,
            "latency_per_frame_ms": latency_per_frame_ms,
            "vision_tokens_ingested": vision_tokens,
            "kv_len_pre_compress": compress_bounds[0],
            "kv_len_post_compress": compress_bounds[1],
            "video_grid_thw": getattr(runner, "last_video_grid_thw", None),
            "tokens_per_frame": runner.token_per_frame,
            "kv_cache_len": kv_len,
            "peak_vram_mb": peak_vram_mb,
            "caption": parsed.caption,
            "score": parsed.score,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"engine": "InfiniPot-V", "chunks": results}, f, indent=2)
    print(f"[+] InfiniPot-V worker completed. Results saved to {args.output}", flush=True)


# ---------------------------------------------------------------------------
# Master Orchestrator
# ---------------------------------------------------------------------------
def print_comparison_table(hermes_data, infinipot_data, jh_data=None):
    """Three-column comparison table: InfiniPot-V | HERMES | JEV+HERMES (optional)."""
    h_chunks  = hermes_data.get("chunks", [])
    i_chunks  = infinipot_data.get("chunks", [])
    jh_chunks = jh_data.get("chunks", []) if jh_data else []
    n = min(len(h_chunks), len(i_chunks))
    three_way = bool(jh_chunks)
    width = 118 if three_way else 90

    print("\n" + "=" * width)
    print("           DEFINITIVE HEAD-TO-HEAD LATENCY COMPARISON REPORT")
    print("=" * width)
    if three_way:
        print(f"{'Chunk':<6} | {'Metric':<22} | {'InfiniPot-V':<15} | {'HERMES':<15} | {'JEV+HERMES':<15} | {'JEV s':<8}")
    else:
        print(f"{'Chunk':<6} | {'Metric':<22} | {'InfiniPot-V':<15} | {'HERMES':<15} | {'Speedup / Diff':<15}")
    print("-" * width)

    n3 = min(n, len(jh_chunks)) if three_way else n
    for idx in range(n3):
        hc = h_chunks[idx]
        ic = i_chunks[idx]

        if three_way:
            jc       = jh_chunks[idx]
            jh_total = jc["total_time_s"]
            jh_net   = jc["net_total_time_s"]
            jh_jev   = jc["jev_time_s"]
            sent     = jc["frames_sent"]
            equiv    = jc["frames_full_equiv"]
            savings  = f"{sent}/{equiv}"
            print(f"#{idx+1:<5} | Total Chunk Latency  | {ic['total_time_s']:>10.3f} s    | {hc['total_time_s']:>10.3f} s    | {jh_total:>10.3f} s    | {jh_jev:>6.3f} s")
            print(f"       | Net HERMES time      |               -     |               -     | {jh_net:>10.3f} s    |")
            print(f"       | Latency per Frame    | {ic['latency_per_frame_ms']:>10.1f} ms   | {hc['latency_per_frame_ms']:>10.1f} ms   | {jc['latency_per_frame_ms']:>10.1f} ms   |")
            print(f"       | - Ingestion Stage    | {ic['ingest_time_s']:>10.3f} s    | {hc['ingest_time_s']:>10.3f} s    | {jc['ingest_time_s']:>10.3f} s    |")
            print(f"       | - Compression Stage  | {ic['compress_time_s']:>10.3f} s    | {hc['compress_time_s']:>10.3f} s    | {jc['compress_time_s']:>10.3f} s    |")
            print(f"       | - Generation Stage   | {ic['gen_time_s']:>10.3f} s    | {hc['gen_time_s']:>10.3f} s    | {jc['gen_time_s']:>10.3f} s    |")
            print(f"       | KV Cache Length      | {ic['kv_cache_len']:>10d} toks | {hc['kv_cache_len']:>10d} toks | {jc['kv_cache_len']:>10d} toks |")
            print(f"       | JEV tier/reason      |               -     |               -     | {jc['jev_tier']}/{jc['jev_reason']:<10}|")
            print(f"       | Frames sent/equiv    |               -     |               -     | {savings:<15}|")
        else:
            tot_diff = hc["total_time_s"] - ic["total_time_s"]
            speedup  = (hc["total_time_s"] / ic["total_time_s"]) if ic["total_time_s"] > 0 else 1.0
            winner   = "InfiniPot-V" if tot_diff > 0 else "HERMES"
            print(f"#{idx+1:<5} | Total Chunk Latency  | {ic['total_time_s']:>10.3f} s    | {hc['total_time_s']:>10.3f} s    | {speedup:.2f}x ({winner})")
            print(f"       | Latency per Frame    | {ic['latency_per_frame_ms']:>10.1f} ms   | {hc['latency_per_frame_ms']:>10.1f} ms   | {hc['latency_per_frame_ms'] - ic['latency_per_frame_ms']:>+7.1f} ms")
            print(f"       | - Ingestion Stage    | {ic['ingest_time_s']:>10.3f} s    | {hc['ingest_time_s']:>10.3f} s    |")
            print(f"       | - Compression Stage  | {ic['compress_time_s']:>10.3f} s    | {hc['compress_time_s']:>10.3f} s    |")
            print(f"       | - Generation Stage   | {ic['gen_time_s']:>10.3f} s    | {hc['gen_time_s']:>10.3f} s    |")
            print(f"       | KV Cache Length      | {ic['kv_cache_len']:>10d} toks | {hc['kv_cache_len']:>10d} toks |")
        print("-" * width)

    # Summary Averages
    mean_h_total    = np.mean([c["total_time_s"] for c in h_chunks])
    mean_i_total    = np.mean([c["total_time_s"] for c in i_chunks])
    mean_h_frame    = np.mean([c["latency_per_frame_ms"] for c in h_chunks])
    mean_i_frame    = np.mean([c["latency_per_frame_ms"] for c in i_chunks])
    mean_h_compress = np.mean([c["compress_time_s"] for c in h_chunks])
    mean_i_compress = np.mean([c["compress_time_s"] for c in i_chunks])
    overall_speedup = mean_h_total / mean_i_total if mean_i_total > 0 else 1.0

    print("\n" + "=" * width)
    print("                               OVERALL SUMMARY")
    print("=" * width)
    print(f"Average Total Chunk Time : InfiniPot-V = {mean_i_total:.3f} s | HERMES = {mean_h_total:.3f} s  (Ratio: {overall_speedup:.2f}x)")
    print(f"Average Latency Per Frame: InfiniPot-V = {mean_i_frame:.1f} ms | HERMES = {mean_h_frame:.1f} ms (Diff: {mean_h_frame - mean_i_frame:+.1f} ms)")
    print(f"Average Compression Time : InfiniPot-V = {mean_i_compress:.3f} s | HERMES = {mean_h_compress:.3f} s")
    if three_way and jh_chunks:
        mean_jh_total = np.mean([c["total_time_s"] for c in jh_chunks])
        mean_jh_net   = np.mean([c["net_total_time_s"] for c in jh_chunks])
        mean_jh_jev   = np.mean([c["jev_time_s"] for c in jh_chunks])
        total_sent    = sum(c["frames_sent"] for c in jh_chunks)
        total_equiv   = sum(c["frames_full_equiv"] for c in jh_chunks)
        saved_pct     = (total_equiv - total_sent) / total_equiv * 100 if total_equiv > 0 else 0.0
        print(f"JEV+HERMES Avg Total   : {mean_jh_total:.3f} s  (Net HERMES: {mean_jh_net:.3f} s | JEV overhead: {mean_jh_jev:.3f} s)")
        print(f"JEV+HERMES Frame Econ  : {total_sent}/{total_equiv} frames sent ({saved_pct:.1f}% fewer)")
    print("=" * width + "\n")


def build_gate_config(gate_config_json, watch_threshold, full_threshold):
    """GateConfig from a thresholds.json / GateConfig JSON; explicit
    --watch-threshold/--full-threshold flags override the JSON when provided."""
    if "vlm_test_ui" not in sys.path:
        sys.path.insert(0, os.path.join(BASE_DIR, "vlm_test_ui"))
    from app.core.jev_gate import GateConfig

    data: dict = {}
    if gate_config_json is not None:
        with open(gate_config_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{gate_config_json}: expected a JSON object")
    if watch_threshold is not None:
        data["route_watch_threshold"] = watch_threshold
    if full_threshold is not None:
        data["route_full_threshold"] = full_threshold
    return GateConfig(**data)


def main():
    parser = argparse.ArgumentParser(description="Isolated Latency Benchmark: HERMES vs. InfiniPot-V")
    parser.add_argument("--mode", choices=["master", "worker"], default="master")
    parser.add_argument("--engine", choices=["hermes", "infinipot", "jev_hermes"])
    parser.add_argument("--jev-model-path", type=str,
                        default=JEV_MODEL_PATH,
                        help="Path to OneJev-0.8B snapshot (jev_hermes engine only)")
    parser.add_argument("--watch-threshold", type=float, default=None,
                        help="Route watch threshold override (jev_hermes engine only)")
    parser.add_argument("--full-threshold", type=float, default=None,
                        help="Route full threshold override (jev_hermes engine only)")
    parser.add_argument("--gate-config-json", type=str, default=None,
                        help="Path to a thresholds.json / GateConfig JSON (jev_hermes engine only; "
                             "explicit --watch-threshold/--full-threshold override it when provided)")
    parser.add_argument("--video", type=str, required=True, help="Path to test MP4 video")
    parser.add_argument("--num-chunks", type=int, default=5, help="Number of sequential chunks to benchmark")
    parser.add_argument("--chunk-s", type=float, default=5.0, help="Chunk duration in seconds")
    parser.add_argument("--overlap-s", type=float, default=1.0, help="Overlap between chunks in seconds")
    parser.add_argument("--vlm-fps", type=float, default=4.0, help="Frames per second to sample per chunk")
    parser.add_argument("--max-dim", type=int, default=448, help="Max image dimension for downsampling")
    parser.add_argument("--kv-size", type=int, default=2000, help="HERMES max KV size")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Max tokens generated per chunk")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to local Qwen model snapshot")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt text")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()
    if args.engine in ("hermes", "infinipot") and (
            args.watch_threshold is not None or args.full_threshold is not None
            or args.gate_config_json is not None):
        parser.error("--watch-threshold/--full-threshold/--gate-config-json are jev_hermes-only flags")
    if args.gate_config_json is not None:
        try:
            build_gate_config(args.gate_config_json, None, None)  # fail fast at the master
        except Exception as exc:
            parser.error(f"--gate-config-json invalid: {exc}")
    args.model_path = ensure_model_path(args.model_path, repo_id="Qwen/Qwen3-VL-4B-Instruct")
    args.jev_model_path = ensure_model_path(args.jev_model_path, repo_id="OmniJev/OneJev-0.8B")

    if args.mode == "worker":
        if args.engine == "hermes":
            run_hermes_worker(args)
        elif args.engine == "infinipot":
            run_infinipot_worker(args)
        elif args.engine == "jev_hermes":
            run_jev_hermes_worker(args)
        return

    # Master Mode: Run both in separate subprocesses to guarantee clean 6GB VRAM
    print("=" * 80)
    print("      ISOLATED STREAMING VLM BENCHMARK: InfiniPot-V vs. HERMES")
    print(f" Video Path : {args.video}")
    print(f" Chunks     : {args.num_chunks} chunks ({args.chunk_s}s per chunk at {args.vlm_fps} fps)")
    print("=" * 80)

    hermes_out    = "hermes_benchmark_telemetry.json"
    infinipot_out = "infinipot_benchmark_telemetry.json"
    jev_hermes_out = "jev_hermes_benchmark_telemetry.json"

    # 1. Run InfiniPot-V
    print("\n>>> [1/2] Launching InfiniPot-V Worker in Clean Subprocess...", flush=True)
    cmd_infinipot = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode", "worker",
        "--engine", "infinipot",
        "--video", args.video,
        "--num-chunks", str(args.num_chunks),
        "--chunk-s", str(args.chunk_s),
        "--overlap-s", str(args.overlap_s),
        "--vlm-fps", str(args.vlm_fps),
        "--max-dim", str(args.max_dim),
        "--max-new-tokens", str(args.max_new_tokens),
        "--model-path", args.model_path,
        "--output", infinipot_out,
    ]
    sub_i = subprocess.run(cmd_infinipot)
    if sub_i.returncode != 0:
        print("[!] InfiniPot-V worker exited with an error.")
        sys.exit(sub_i.returncode)

    # Cooldown between runs
    time.sleep(3)

    # 2. Run HERMES
    print("\n>>> [2/2] Launching HERMES Worker in Clean Subprocess...", flush=True)
    cmd_hermes = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode", "worker",
        "--engine", "hermes",
        "--video", args.video,
        "--num-chunks", str(args.num_chunks),
        "--chunk-s", str(args.chunk_s),
        "--overlap-s", str(args.overlap_s),
        "--vlm-fps", str(args.vlm_fps),
        "--max-dim", str(args.max_dim),
        "--kv-size", str(args.kv_size),
        "--max-new-tokens", str(args.max_new_tokens),
        "--model-path", args.model_path,
        "--output", hermes_out,
    ]
    sub_h = subprocess.run(cmd_hermes)
    if sub_h.returncode != 0:
        print("[!] HERMES worker exited with an error.")
        sys.exit(sub_h.returncode)

    # Cooldown before JEV+HERMES run
    time.sleep(3)

    # 3. Run JEV+HERMES
    print("\n>>> [3/3] Launching JEV+HERMES Worker in Clean Subprocess...", flush=True)
    cmd_jh = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode", "worker",
        "--engine", "jev_hermes",
        "--video", args.video,
        "--num-chunks", str(args.num_chunks),
        "--chunk-s", str(args.chunk_s),
        "--overlap-s", str(args.overlap_s),
        "--vlm-fps", str(args.vlm_fps),
        "--max-dim", str(args.max_dim),
        "--kv-size", str(args.kv_size),
        "--max-new-tokens", str(args.max_new_tokens),
        "--model-path", args.model_path,
        "--jev-model-path", args.jev_model_path,
        "--output", jev_hermes_out,
    ]
    if args.gate_config_json is not None:
        cmd_jh += ["--gate-config-json", args.gate_config_json]
    if args.watch_threshold is not None:
        cmd_jh += ["--watch-threshold", str(args.watch_threshold)]
    if args.full_threshold is not None:
        cmd_jh += ["--full-threshold", str(args.full_threshold)]
    sub_jh = subprocess.run(cmd_jh)
    if sub_jh.returncode != 0:
        print("[!] JEV+HERMES worker exited with an error.")
        sys.exit(sub_jh.returncode)

    # Load results & display three-way comparison
    if os.path.exists(infinipot_out) and os.path.exists(hermes_out):
        with open(infinipot_out, "r", encoding="utf-8") as fi:
            i_data = json.load(fi)
        with open(hermes_out, "r", encoding="utf-8") as fh:
            h_data = json.load(fh)
        jh_data = None
        if os.path.exists(jev_hermes_out):
            with open(jev_hermes_out, "r", encoding="utf-8") as fjh:
                jh_data = json.load(fjh)
        print_comparison_table(h_data, i_data, jh_data)
    else:
        print("[!] Could not find telemetry JSON files to build report.")


if __name__ == "__main__":
    main()
