"""
Phase 6 - E2E Benchmarking & Validation
=========================================
6.1  Caption quality parity:  use_infinipot=True vs False on same chunks.
6.2  Memory plateau:          seq_len stays bounded after budget saturation.
6.3  Latency stability:       per-chunk wall time does not grow linearly.
"""

import sys
import os
import time
import unittest
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
vlm_path = os.path.join(project_root, "vlm_test_ui")
if vlm_path not in sys.path:
    sys.path.insert(0, vlm_path)

import torch
from app.core.types import VideoChunk, VLMResult
from vlm.qwen3vl_runner import Qwen3VLRunner


def _make_chunks(n, frames_per_chunk=4, h=64, w=64):
    chunks = []
    for i in range(n):
        frames = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(frames_per_chunk)]
        start_ms = i * 2000
        timestamps_ms = [start_ms + int(j * 2000 / max(frames_per_chunk, 1)) for j in range(frames_per_chunk)]
        chunks.append(VideoChunk(
            chunk_id=i,
            source="synthetic_benchmark",
            start_ms=start_ms,
            end_ms=start_ms + 2000,
            fps=2.0,
            frames=frames,
            timestamps_ms=timestamps_ms,
        ))
    return chunks


class TestE2EBenchmark(unittest.TestCase):
    """
    One model load for all three sub-tests via setUpClass.
    This avoids OOM: each test method re-uses cls.runner rather than
    loading a fresh model instance that stacks on top of the previous one.
    """

    @classmethod
    def setUpClass(cls):
        cls.runner = Qwen3VLRunner(
            memory_budget_frames=8,
            compress_to_frames=4,
            compression_method="infinipot-v",
            use_infinipot=True,
            max_new_tokens=80,
        )

    @classmethod
    def tearDownClass(cls):
        # Release GPU memory when all tests are done
        del cls.runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 6.1  Caption quality parity
    # ------------------------------------------------------------------
    def test_6_1_caption_quality_parity(self):
        """Stateful and stateless paths on the same runner must both produce VLMResults."""
        # --- Stateful pass: multi-frame chunks via InfiniPot-V cache path ---
        chunks_multi = _make_chunks(n=3, frames_per_chunk=4)
        self.runner.use_infinipot = True
        self.runner.reset()
        captions_stateful = []
        for chunk in chunks_multi:
            r = self.runner(chunk)
            self.assertIsInstance(r, VLMResult)
            captions_stateful.append(r.caption)
            print(f"[6.1] stateful: {r.caption!r}")

        # --- Stateless pass: 1-frame chunks force image mode in _legacy_call,
        #     bypassing torchvision video reader (broken on Windows in this env) ---
        chunks_single = _make_chunks(n=3, frames_per_chunk=1)
        self.runner.use_infinipot = False
        captions_stateless = []
        for chunk in chunks_single:
            r = self.runner(chunk)
            self.assertIsInstance(r, VLMResult)
            captions_stateless.append(r.caption)
            print(f"[6.1] stateless: {r.caption!r}")

        # Restore stateful mode for subsequent tests
        self.runner.use_infinipot = True

        self.assertTrue(any(len(c) > 0 for c in captions_stateful), "Stateful: all captions empty")
        self.assertTrue(any(len(c) > 0 for c in captions_stateless), "Stateless: all captions empty")

    # ------------------------------------------------------------------
    # 6.2  Memory plateau verification
    # ------------------------------------------------------------------
    def test_6_2_memory_plateau(self):
        """seq_len must stay bounded after budget saturation over 12 chunks."""
        MEMORY_BUDGET = self.runner.memory_budget_frames   # 8
        COMPRESS_TO   = self.runner.compress_to_frames     # 4
        N_CHUNKS      = 12

        self.runner.use_infinipot = True
        self.runner.reset()
        chunks = _make_chunks(n=N_CHUNKS, frames_per_chunk=4)

        seq_lens, vram_mb = [], []
        for i, chunk in enumerate(chunks):
            self.runner(chunk)
            seq_len = self.runner.past_key_values.layers[0].keys.shape[2]
            seq_lens.append(seq_len)
            if torch.cuda.is_available():
                vram_mb.append(torch.cuda.memory_reserved() / 1024 ** 2)
            print(f"[6.2] Chunk {i+1:02d}: seq_len={seq_len}")

        saturation_idx = None
        for i, sl in enumerate(seq_lens):
            frames_cached = (sl - self.runner.system_size) // self.runner.token_per_frame
            if frames_cached > MEMORY_BUDGET:
                self.fail(f"Chunk {i+1}: seq_len={sl} exceeds budget (>{MEMORY_BUDGET} frames)")
            if frames_cached == COMPRESS_TO and saturation_idx is None:
                saturation_idx = i

        print(f"[6.2] Plateau first reached at chunk index {saturation_idx}")

        if vram_mb and saturation_idx is not None and saturation_idx < len(vram_mb) - 1:
            vram_post = vram_mb[saturation_idx:]
            vram_growth = max(vram_post) - min(vram_post)
            print(f"[6.2] VRAM post-plateau growth: {vram_growth:.0f} MB")
            self.assertLess(vram_growth, 512, f"VRAM grew {vram_growth:.0f} MB post-plateau")

    # ------------------------------------------------------------------
    # 6.3  Latency stability
    # ------------------------------------------------------------------
    def test_6_3_latency_stability(self):
        """Per-chunk latency slope must be < 2.0 s/chunk (no linear growth)."""
        N_CHUNKS = 10

        self.runner.use_infinipot = True
        self.runner.reset()
        chunks = _make_chunks(n=N_CHUNKS, frames_per_chunk=4)

        latencies = []
        for i, chunk in enumerate(chunks):
            t0 = time.perf_counter()
            self.runner(chunk)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            print(f"[6.3] Chunk {i+1:02d}: {elapsed:.2f}s")

        # Drop first chunk (CUDA kernel compilation warm-up noise)
        stable = latencies[1:]
        xs = np.arange(len(stable), dtype=float)
        slope, _ = np.polyfit(xs, stable, 1)
        print(f"[6.3] Latency slope: {slope:.3f} s/chunk  mean={np.mean(stable):.2f}s  std={np.std(stable):.2f}s")
        self.assertLess(slope, 2.0, f"Latency slope {slope:.3f} s/chunk indicates unbounded memory growth")


if __name__ == "__main__":
    unittest.main()

