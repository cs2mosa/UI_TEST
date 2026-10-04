"""
RoPE / KV-cache reindex utilities adapted from HERMES/inference/reindex_3d.py
for Qwen3-VL-4B-Instruct (interleaved 3D M-RoPE, mrope_section=[24,20,20]).

The only change from the original is:
  - Import apply_multimodal_rotary_pos_emb from qwen3_vl instead of qwen2_5_vl
  - Default mrope_section changed to (24, 20, 20)
  - _get_rotary_module handles Qwen3-VL model tree
  - hidden_size default changed to 2560 (Qwen3-VL-4B)
"""
import torch
from typing import Tuple
from transformers import DynamicCache

# transformers 5.17: apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
# Interleaved MRoPE is handled internally by Qwen3VLTextRotaryEmbedding.recomposition_frequencies
# so apply_rotary_pos_emb needs no mrope_section argument.
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb


# ---------------------------------------------------------------------------
# DynamicCache compatibility shim for Transformers 5.17+
# ---------------------------------------------------------------------------
if not hasattr(DynamicCache, "__getitem__"):
    def _dynamic_cache_getitem(self, layer_idx):
        layer = self.layers[layer_idx]
        return (layer.keys, layer.values)
    DynamicCache.__getitem__ = _dynamic_cache_getitem

def _dynamic_cache_iter(self):
    for layer in self.layers:
        yield (layer.keys, layer.values)
DynamicCache.__iter__ = _dynamic_cache_iter

if not hasattr(DynamicCache, "__len__"):
    def _dynamic_cache_len(self):
        return len(self.layers)
    DynamicCache.__len__ = _dynamic_cache_len

if not hasattr(DynamicCache, "from_legacy_cache"):
    @classmethod
    def _from_legacy_cache(cls, past_key_values):
        cache = cls()
        if past_key_values is not None:
            for layer_idx, (k, v) in enumerate(past_key_values):
                cache.update(k, v, layer_idx)
        return cache
    DynamicCache.from_legacy_cache = _from_legacy_cache


def get_cache_seq_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, DynamicCache):
        return past_key_values.get_seq_length()
    return past_key_values[0][0].shape[2]


def contiguous_kv(past_key_values):
    if isinstance(past_key_values, DynamicCache):
        for layer in past_key_values.layers:
            if hasattr(layer, "keys") and layer.keys is not None:
                layer.keys = layer.keys.contiguous()
            if hasattr(layer, "values") and layer.values is not None:
                layer.values = layer.values.contiguous()
        return past_key_values
    new_legacy = []
    for (k_layer, v_layer) in past_key_values:
        new_legacy.append((k_layer.contiguous(), v_layer.contiguous()))
    return new_legacy


def _get_rotary_module(llm) -> torch.nn.Module:
    """Find rotary embedding module on a Qwen3-VL language model."""
    if hasattr(llm, "rotary_emb"):
        return llm.rotary_emb
    if hasattr(llm, "model") and hasattr(llm.model, "rotary_emb"):
        return llm.model.rotary_emb
    if hasattr(llm, "layers") and len(llm.layers) > 0:
        if hasattr(llm.layers[0], "self_attn") and hasattr(llm.layers[0].self_attn, "rotary_emb"):
            return llm.layers[0].self_attn.rotary_emb
    if hasattr(llm, "model") and hasattr(llm.model, "layers") and len(llm.model.layers) > 0:
        if hasattr(llm.model.layers[0], "self_attn") and hasattr(
            llm.model.layers[0].self_attn, "rotary_emb"
        ):
            return llm.model.layers[0].self_attn.rotary_emb
    raise AttributeError("Cannot find rotary_emb module on language_model")


def _get_mrope_section(llm) -> Tuple[int, int, int]:
    """Extract mrope_section from Qwen3-VL config. Default: (24, 20, 20)."""
    cfg = getattr(llm, "config", None)
    if cfg is None:
        return (24, 20, 20)
    # Qwen3-VL stores it under text_config.rope_scaling
    text_cfg = getattr(cfg, "text_config", None)
    if text_cfg and getattr(text_cfg, "rope_scaling", None):
        sec = text_cfg.rope_scaling.get("mrope_section", None)
        if isinstance(sec, (list, tuple)) and len(sec) == 3:
            return tuple(sec)
    # Fallback: top-level rope_scaling
    rope_scaling = getattr(cfg, "rope_scaling", None)
    if rope_scaling and isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling:
        sec = rope_scaling["mrope_section"]
        if isinstance(sec, (list, tuple)) and len(sec) == 3:
            return tuple(sec)
    return (24, 20, 20)


def compute_cos_sin_for_positions(
    llm,
    seq_len: int,
    position_ids_3d: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    Compute cos/sin for given 3D positions using interleaved M-RoPE (Qwen3-VL).

    Args:
        llm: language model or model with rotary_emb
        seq_len: sequence length
        position_ids_3d: shape [3, seq_len] or [3, 1, seq_len]
        dtype: target dtype
        device: target device

    Returns:
        cos, sin suitable for apply_multimodal_rotary_pos_emb
    """
    rotary_emb = _get_rotary_module(llm)

    # Resolve hidden_size from Qwen3-VL config tree
    hidden_size = None
    cfg = getattr(llm, "config", None)
    if cfg is not None:
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            hidden_size = getattr(text_cfg, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(cfg, "hidden_size", None)
    if hidden_size is None:
        hidden_size = 2560  # Qwen3-VL-4B default

    # Ensure [3, 1, seq_len] format
    if position_ids_3d.dim() == 2:
        position_ids_3d = position_ids_3d.unsqueeze(1)

    pos = position_ids_3d.to(device)
    dummy_h = torch.zeros((1, seq_len, hidden_size), device=device, dtype=dtype)
    cos, sin = rotary_emb(dummy_h, pos)
    cos = cos.to(dtype)
    sin = sin.to(dtype)
    return cos, sin


def rotary_delta(cos_old, sin_old, cos_new, sin_new):
    """Compute rotation delta: cos(a-b), sin(a-b)."""
    cos_delta = cos_new * cos_old + sin_new * sin_old
    sin_delta = sin_new * cos_old - cos_new * sin_old
    return cos_delta, sin_delta


def apply_rotary_delta_to_keys_only(
    key_states: torch.Tensor,
    cos_delta,
    sin_delta,
    mrope_section=None,  # unused in transformers 5.17 — kept for API compatibility
) -> torch.Tensor:
    """
    Apply rotary position delta to key states using Qwen3-VL's apply_rotary_pos_emb.

    In transformers 5.17, apply_rotary_pos_emb(q, k, cos, sin) applies the rotation
    directly; no mrope_section argument is required — interleaved MRoPE is already
    baked into the cos/sin returned by Qwen3VLTextRotaryEmbedding.
    We call it with key_states as both q and k, then discard the q result.
    """
    _q_rot, k_rot = apply_rotary_pos_emb(
        key_states,   # dummy query (result discarded)
        key_states,
        cos_delta,
        sin_delta,
    )
    return k_rot
