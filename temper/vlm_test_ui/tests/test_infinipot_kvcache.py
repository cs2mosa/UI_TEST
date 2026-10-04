"""
Unit tests for InfiniPot-V KV-cache compression in isolation (using synthetic cache).
Does not require loading a real model or GPU.
"""

import sys
import os
import torch
import unittest
from types import SimpleNamespace
from transformers.cache_utils import DynamicCache

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from vlm_test_ui.vlm.infinipot_v.kvcache_utils import process_kv_cache


def create_synthetic_cache(
    num_layers: int = 4,
    bsz: int = 1,
    num_heads: int = 8,
    system_size: int = 20,
    total_frames: int = 8,
    token_per_frame: int = 16,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
):
    """Construct a synthetic DynamicCache with known dimensions and prefix values."""
    cache = DynamicCache()
    total_seq_len = system_size + total_frames * token_per_frame
    
    original_keys = []
    original_values = []

    for layer_idx in range(num_layers):
        # Fill system tokens with distinct values so byte-identity check is rigorous
        k = torch.randn(bsz, num_heads, total_seq_len, head_dim, dtype=dtype, device=device)
        v = torch.randn(bsz, num_heads, total_seq_len, head_dim, dtype=dtype, device=device)
        
        # Give system tokens fixed deterministic values
        k[:, :, :system_size, :] = (layer_idx + 1) * 1.5
        v[:, :, :system_size, :] = (layer_idx + 1) * 2.5
        
        cache.update(k.clone(), v.clone(), layer_idx)
        original_keys.append(k.clone())
        original_values.append(v.clone())

    mock_model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(
                layers=[SimpleNamespace() for _ in range(num_layers)]
            )
        ),
        config=SimpleNamespace(
            height_width=4,  # 4x4 = 16 tokens per frame
            num_hidden_layers=num_layers,
        )
    )

    return cache, mock_model, original_keys, original_values


class TestInfiniPotKVCache(unittest.TestCase):
    def _run_method_test(self, method: str):
        num_layers = 4
        system_size = 20
        total_frames = 8
        compress_frame_num = 4
        token_per_frame = 16
        expected_seq_len = system_size + compress_frame_num * token_per_frame  # 20 + 64 = 84

        cache, mock_model, orig_k, orig_v = create_synthetic_cache(
            num_layers=num_layers,
            system_size=system_size,
            total_frames=total_frames,
            token_per_frame=token_per_frame,
        )

        compressed_cache, _ = process_kv_cache(
            past_key_values=cache,
            model=mock_model,
            system_size=system_size,
            inst_size=0,
            token_per_frame=token_per_frame,
            compress_frame_num=compress_frame_num,
            method=method,
        )

        # 1. Assert resulting sequence length equals system_size + compress_frame_num * token_per_frame
        for layer_idx in range(num_layers):
            layer_k = compressed_cache.layers[layer_idx].keys
            layer_v = compressed_cache.layers[layer_idx].values

            self.assertEqual(
                layer_k.shape[2], expected_seq_len,
                f"Layer {layer_idx} keys seq_len {layer_k.shape[2]} != expected {expected_seq_len}"
            )
            self.assertEqual(
                layer_v.shape[2], expected_seq_len,
                f"Layer {layer_idx} values seq_len {layer_v.shape[2]} != expected {expected_seq_len}"
            )

            # Assert bfloat16 preservation through compression
            self.assertEqual(
                layer_k.dtype, torch.bfloat16,
                f"Layer {layer_idx} keys dtype changed from bfloat16 to {layer_k.dtype} during {method}"
            )
            self.assertEqual(
                layer_v.dtype, torch.bfloat16,
                f"Layer {layer_idx} values dtype changed from bfloat16 to {layer_v.dtype} during {method}"
            )

            # 2. Assert system-token prefix is byte-identical before and after compression
            self.assertTrue(
                torch.equal(layer_k[:, :, :system_size, :], orig_k[layer_idx][:, :, :system_size, :]),
                f"Layer {layer_idx} system-token prefix in keys was modified during {method} compression!"
            )
            self.assertTrue(
                torch.equal(layer_v[:, :, :system_size, :], orig_v[layer_idx][:, :, :system_size, :]),
                f"Layer {layer_idx} system-token prefix in values was modified during {method} compression!"
            )

    def test_uniform_method(self):
        self._run_method_test("uniform")

    def test_swa_method(self):
        self._run_method_test("swa")

    def test_infinipot_v_method(self):
        self._run_method_test("infinipot-v")

    def test_infinipot_v_adaptive_pooling(self):
        """Verify that infinipot-v with adaptive pooling enabled runs and compresses correctly."""
        num_layers = 4
        system_size = 10
        total_frames = 6
        compress_frame_num = 3
        token_per_frame = 16
        expected_seq_len = system_size + compress_frame_num * token_per_frame

        cache, mock_model, orig_k, orig_v = create_synthetic_cache(
            num_layers=num_layers,
            system_size=system_size,
            total_frames=total_frames,
            token_per_frame=token_per_frame,
        )

        compressed_cache, _ = process_kv_cache(
            past_key_values=cache,
            model=mock_model,
            system_size=system_size,
            inst_size=0,
            token_per_frame=token_per_frame,
            compress_frame_num=compress_frame_num,
            method="infinipot-v",
            adaptive_pooling=True,
        )

        for layer_idx in range(num_layers):
            layer_k = compressed_cache.layers[layer_idx].keys
            layer_v = compressed_cache.layers[layer_idx].values
            self.assertEqual(layer_k.shape[2], expected_seq_len)
            self.assertEqual(layer_k.dtype, torch.bfloat16)
            self.assertEqual(layer_v.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(layer_k[:, :, :system_size, :], orig_k[layer_idx][:, :, :system_size, :]))

    def test_no_op_when_under_budget(self):
        """When current_frame_num <= compress_frame_num, it should be an exact no-op."""
        system_size = 15
        total_frames = 4
        compress_frame_num = 4  # exactly equal, no compression needed
        token_per_frame = 16

        cache, mock_model, orig_k, orig_v = create_synthetic_cache(
            num_layers=2,
            system_size=system_size,
            total_frames=total_frames,
            token_per_frame=token_per_frame,
        )

        compressed_cache, _ = process_kv_cache(
            past_key_values=cache,
            model=mock_model,
            system_size=system_size,
            inst_size=0,
            token_per_frame=token_per_frame,
            compress_frame_num=compress_frame_num,
            method="uniform",
        )

        for layer_idx in range(2):
            self.assertEqual(compressed_cache.layers[layer_idx].keys.dtype, torch.bfloat16)
            self.assertEqual(compressed_cache.layers[layer_idx].values.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(compressed_cache.layers[layer_idx].keys, orig_k[layer_idx]))
            self.assertTrue(torch.equal(compressed_cache.layers[layer_idx].values, orig_v[layer_idx]))


if __name__ == "__main__":
    unittest.main()
