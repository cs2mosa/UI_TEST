"""
Unit tests for hermes_reindex.py — no GPU required.
"""
import unittest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

from vlm.hermes_reindex import (
    get_cache_seq_len,
    contiguous_kv,
    rotary_delta,
    _get_mrope_section,
)


class TestGetCacheSeqLen(unittest.TestCase):
    def test_none_returns_zero(self):
        self.assertEqual(get_cache_seq_len(None), 0)

    def test_list_of_tuples(self):
        k = torch.zeros(1, 8, 100, 128)
        v = torch.zeros(1, 8, 100, 128)
        cache = [(k, v)] * 4
        self.assertEqual(get_cache_seq_len(cache), 100)

    def test_single_layer_cache(self):
        k = torch.zeros(1, 8, 37, 128)
        v = torch.zeros(1, 8, 37, 128)
        cache = [(k, v)]
        self.assertEqual(get_cache_seq_len(cache), 37)


class TestContiguousKV(unittest.TestCase):
    def test_returns_contiguous(self):
        k = torch.zeros(1, 8, 10, 64)
        v = torch.zeros(1, 8, 10, 64)
        # Make non-contiguous by transposing then transposing back
        k_noncontig = k.transpose(2, 3).transpose(2, 3)
        v_noncontig = v.transpose(2, 3).transpose(2, 3)
        cache = [(k_noncontig, v_noncontig)]
        result = contiguous_kv(cache)
        for (k_r, v_r) in result:
            self.assertTrue(k_r.is_contiguous())
            self.assertTrue(v_r.is_contiguous())


class TestRotaryDelta(unittest.TestCase):
    def test_identity_delta(self):
        cos = torch.ones(1, 10, 64)
        sin = torch.zeros(1, 10, 64)
        cos_d, sin_d = rotary_delta(cos, sin, cos, sin)
        # cos(a - a) = 1, sin(a - a) = 0
        self.assertTrue(torch.allclose(cos_d, torch.ones_like(cos_d), atol=1e-5))
        self.assertTrue(torch.allclose(sin_d, torch.zeros_like(sin_d), atol=1e-5))

    def test_90_degree_delta(self):
        # cos(0) = 1, sin(0) = 0 → cos(90-0)=0, sin(90-0)=1
        cos_old = torch.ones(1, 1, 64)
        sin_old = torch.zeros(1, 1, 64)
        cos_new = torch.zeros(1, 1, 64)
        sin_new = torch.ones(1, 1, 64)
        cos_d, sin_d = rotary_delta(cos_old, sin_old, cos_new, sin_new)
        self.assertTrue(torch.allclose(cos_d, torch.zeros_like(cos_d), atol=1e-5))
        self.assertTrue(torch.allclose(sin_d, torch.ones_like(sin_d), atol=1e-5))


class TestGetMropeSection(unittest.TestCase):
    def test_qwen3vl_config_structure(self):
        """Should extract [24,20,20] from Qwen3-VL config layout."""
        text_config = SimpleNamespace(
            rope_scaling={"mrope_section": [24, 20, 20], "mrope_interleaved": True}
        )
        mock_model = MagicMock()
        mock_model.config = SimpleNamespace(text_config=text_config)
        result = _get_mrope_section(mock_model)
        self.assertEqual(result, (24, 20, 20))

    def test_fallback_when_config_missing(self):
        mock_model = MagicMock()
        mock_model.config = None
        result = _get_mrope_section(mock_model)
        self.assertEqual(result, (24, 20, 20))

    def test_top_level_rope_scaling(self):
        """Also accept rope_scaling at the top config level."""
        mock_model = MagicMock()
        mock_model.config = SimpleNamespace(
            text_config=None,
            rope_scaling={"mrope_section": [16, 24, 24]},
        )
        result = _get_mrope_section(mock_model)
        self.assertEqual(result, (16, 24, 24))


if __name__ == "__main__":
    unittest.main()
