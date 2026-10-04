"""
HERMES hierarchical KV-cache engine adapted for Qwen3-VL-4B-Instruct.

Architecture differences from the original HERMES qwenvl_hermes.py:
  - Model class:  Qwen3VLForConditionalGeneration  (vs Qwen2_5_VL...)
  - RoPE layout:  interleaved MRoPE, mrope_section=[24,20,20], mrope_interleaved=True
  - Vision fusion: DeepStack (handled transparently by get_video_features)
  - Num layers:   36  (vs 28 for Qwen2.5-VL-3B/7B)
  - hidden_size:  2560 (vs 3584 for Qwen2.5-VL-7B)
  - Quantization: 4-bit BitsAndBytes for 6 GB VRAM GPUs (default)

This file is a near-verbatim port of HERMES/inference/qwenvl_hermes.py with
the above adaptations applied.  Method names and logic are identical so that
the rest of the HERMES ecosystem (hermes_runner.py) can call the same API.
"""
from __future__ import annotations

import re
import time
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, DynamicCache
from transformers import BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Transformers 5.17 DynamicCache backward-compatibility shim
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
# transformers 5.17: apply_rotary_pos_emb(q, k, cos, sin) — no mrope_section arg needed
# Interleaved MRoPE is handled by Qwen3VLTextRotaryEmbedding.recomposition_frequencies internally
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

# Apply FA patch before any model forward passes
import vlm.hermes_fa_patch  # noqa: F401  side-effect import

from vlm.hermes_base import AbstractHermes
from vlm.vision_common import preprocess_video_frames
from vlm.hermes_reindex import (
    get_cache_seq_len,
    contiguous_kv,
    _get_rotary_module,
    _get_mrope_section,
    compute_cos_sin_for_positions,
    rotary_delta,
    apply_rotary_delta_to_keys_only,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def get_qwen3_vl_position_ids(
    video_grid_thw,
    seq_len: int,
    offset: int = 0,
    spatial_merge_size: int = 2,
    vision_config=None,
    sample_fps: float = 1.0,
) -> torch.Tensor:
    """
    Compute 3D position IDs for a video chunk (T, H, W).
    Mirrors get_qwen2_5_vl_position_ids from HERMES qwenvl_hermes.py.
    """
    t, h, w = video_grid_thw
    llm_grid_t = t
    llm_grid_h = h // spatial_merge_size
    llm_grid_w = w // spatial_merge_size

    if vision_config is not None:
        spatial_merge_size = getattr(vision_config, "spatial_merge_size", spatial_merge_size)
        llm_grid_h = h // spatial_merge_size
        llm_grid_w = w // spatial_merge_size

    if llm_grid_t * llm_grid_h * llm_grid_w != seq_len:
        if t * h * w == seq_len:
            llm_grid_t, llm_grid_h, llm_grid_w = t, h, w
        else:
            print(
                f"Warning: Grid {t}x{h}x{w} (merged: {llm_grid_t}x{llm_grid_h}x{llm_grid_w})"
                f" does not match seq_len {seq_len}"
            )

    if vision_config is not None and hasattr(vision_config, "tokens_per_second"):
        tokens_per_second = vision_config.tokens_per_second
        second_per_grid_t = 2 / sample_fps
        range_tensor = torch.arange(llm_grid_t, dtype=torch.float32).view(-1, 1)
        expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
        time_tensor = expanded_range * second_per_grid_t * tokens_per_second
        t_index = time_tensor.flatten() + offset
    else:
        t_index = (
            torch.arange(llm_grid_t, dtype=torch.float32)
            .view(-1, 1)
            .expand(-1, llm_grid_h * llm_grid_w)
            .flatten()
            + offset
        )

    h_index = (
        torch.arange(llm_grid_h, dtype=torch.float32)
        .view(1, -1, 1)
        .expand(llm_grid_t, -1, llm_grid_w)
        .flatten()
        + offset
    )
    w_index = (
        torch.arange(llm_grid_w, dtype=torch.float32)
        .view(1, 1, -1)
        .expand(llm_grid_t, llm_grid_h, -1)
        .flatten()
        + offset
    )

    return torch.stack([t_index, h_index, w_index])


# ---------------------------------------------------------------------------
# Main HERMES engine class
# ---------------------------------------------------------------------------

class Qwen3VLHermes(Qwen3VLForConditionalGeneration, AbstractHermes):
    """
    Qwen3-VL-4B with HERMES strict-shrink KV-cache support.
    Uses interleaved 3D M-RoPE and inter-layer consistency optimisation.
    """

    def __init__(self, config, processor, init_prompt_ids, kv_size, streaming=True, sample_fps=1.0):
        AbstractHermes.__init__(self, processor, init_prompt_ids, kv_size)
        self.streaming = streaming
        self.sample_fps = sample_fps

        num_layers = getattr(config, "num_hidden_layers", None)
        if num_layers is None:
            text_cfg = getattr(config, "text_config", None)
            num_layers = getattr(text_cfg, "num_hidden_layers", 36) if text_cfg else 36
        self.num_layers = num_layers

        self.short_term_ratio = 0.1
        self.long_term_ratio = 0.3
        self.short_term_threshold = int(self.num_layers * self.short_term_ratio)
        self.long_term_threshold = int(self.num_layers * (1 - self.long_term_ratio))

        self._position_ids_cache = [None for _ in range(num_layers)]
        self._layer_position_ids = {}
        self._hook_handles = []
        self.total_processed_frames = 0
        self.last_video_grid_thw = None

        self._mrope_section = _get_mrope_section(self.model.language_model)
        self._register_forward_hooks()

    # ------------------------------------------------------------------
    # Transformers 5.17 attribute bridge
    # ------------------------------------------------------------------

    @property
    def language_model(self):
        """
        In transformers 5.17, Qwen3VLForConditionalGeneration stores the
        multimodal wrapper as self.model (Qwen3VLModel), which in turn
        stores the text-only transformer as self.model.language_model.
        This property exposes it directly so HERMES code can use
        self.language_model.layers, self.language_model.config, etc.
        """
        return self.model.language_model

    # ------------------------------------------------------------------
    # KV-cache helpers
    # ------------------------------------------------------------------

    def _ensure_dynamic_cache(self):
        if self.kv_cache is None:
            return
        if not isinstance(self.kv_cache, DynamicCache):
            self.kv_cache = DynamicCache.from_legacy_cache(self.kv_cache)

    def _sanitize_keep_indices(self, keep_indices_1d: torch.Tensor, seq_len: int) -> torch.Tensor:
        keep_indices_1d = keep_indices_1d.to(self.device)
        keep_indices_1d = keep_indices_1d[(keep_indices_1d >= 0) & (keep_indices_1d < seq_len)]
        if keep_indices_1d.numel() == 0:
            return torch.tensor([0], device=self.device)
        return torch.unique(keep_indices_1d, sorted=True)

    def _get_cache_seq_len_per_layer(self) -> list:
        if self.kv_cache is None:
            return [0] * self.num_layers
        lengths = []
        for layer_idx in range(len(self.kv_cache)):
            k_layer, v_layer = self.kv_cache[layer_idx]
            lengths.append(k_layer.shape[2])
        return lengths

    def _get_next_global_offset_per_layer(self) -> list:
        offsets = []
        for layer_idx in range(self.num_layers):
            cache = self._position_ids_cache[layer_idx]
            if cache is not None and cache.numel() > 0:
                offsets.append(int(cache.max().item()) + 1)
            else:
                offsets.append(0)
        return offsets

    def _truncate_kv_cache(self, target_lengths):
        if self.kv_cache is None:
            return
        truncated_cache = []
        for layer_idx, (k_cache, v_cache) in enumerate(self.kv_cache):
            target_len = target_lengths if isinstance(target_lengths, int) else target_lengths[layer_idx]
            truncated_cache.append((k_cache[:, :, :target_len, :], v_cache[:, :, :target_len, :]))
        if isinstance(self.kv_cache, DynamicCache):
            self.kv_cache = DynamicCache.from_legacy_cache(truncated_cache)
        else:
            self.kv_cache = truncated_cache

    # ------------------------------------------------------------------
    # Forward hook management (layer-specific position IDs)
    # ------------------------------------------------------------------

    def _register_forward_hooks(self):
        def make_hook(layer_idx):
            def hook(module, args, kwargs):
                if layer_idx in self._layer_position_ids:
                    kwargs["position_ids"] = self._layer_position_ids[layer_idx]
                return args, kwargs
            return hook

        for layer_idx, layer in enumerate(self.language_model.layers):
            handle = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
            self._hook_handles.append(handle)

    def _clear_forward_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    # ------------------------------------------------------------------
    # Position ID bookkeeping
    # ------------------------------------------------------------------

    def _append_position_ids_layer(self, layer_idx: int, start_per_dim: list, length: int):
        device = self.device
        new_pos = torch.zeros((3, length), device=device, dtype=torch.float32)
        for dim in range(3):
            new_pos[dim, :] = torch.arange(
                start_per_dim[dim], start_per_dim[dim] + length,
                device=device, dtype=torch.float32,
            )
        cache = self._position_ids_cache[layer_idx]
        if cache is None:
            self._position_ids_cache[layer_idx] = new_pos
        else:
            self._position_ids_cache[layer_idx] = torch.cat([cache, new_pos], dim=1)

    def _append_position_ids_layer_explicit(self, layer_idx: int, new_pos: torch.Tensor):
        cache = self._position_ids_cache[layer_idx]
        if cache is None:
            self._position_ids_cache[layer_idx] = new_pos.to(self.device)
        else:
            self._position_ids_cache[layer_idx] = torch.cat(
                [cache, new_pos.to(self.device)], dim=1
            )

    def _build_position_ids_3d_for_text(self, global_offset: int, q_len: int, batch: int) -> torch.Tensor:
        """[3, batch, q_len] — text tokens use identical position across all 3 dims."""
        pos_1d = torch.arange(global_offset, global_offset + q_len, device=self.device, dtype=torch.float32)
        pos_2d = pos_1d.unsqueeze(0).expand(batch, -1)
        return pos_2d.unsqueeze(0).expand(3, -1, -1).clone()

    def _build_position_ids_3d_for_vision(self, grid_pos_ids: torch.Tensor, batch: int) -> torch.Tensor:
        """[3, batch, q_len] — vision tokens use grid-based 3D positions."""
        return grid_pos_ids.unsqueeze(1).expand(3, batch, -1).clone()

    # ------------------------------------------------------------------
    # Encode init prompt
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor(self.init_prompt_ids, device=self.device)

        seq_len = self.init_prompt_ids.shape[-1]
        pos_1d = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        position_ids_3d = pos_1d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1).clone()

        output = self.language_model(
            input_ids=self.init_prompt_ids,
            use_cache=True,
            return_dict=True,
            position_ids=position_ids_3d,
        )
        self.kv_cache = output.past_key_values
        self.visual_start_idx = self.kv_cache[0][0].shape[2]

        self._ensure_dynamic_cache()
        self.total_processed_frames = 0

        curr_lens = self._get_cache_seq_len_per_layer()
        for layer_idx in range(self.num_layers):
            pos = torch.arange(curr_lens[layer_idx], device=self.device, dtype=torch.float32)
            self._position_ids_cache[layer_idx] = pos.unsqueeze(0).expand(3, -1).clone()

    # ------------------------------------------------------------------
    # Encode video chunk
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_frames(video_chunk: torch.Tensor, max_side: int = 448) -> torch.Tensor:
        """
        Resize frames so neither side exceeds max_side pixels.
        Expects [N, H, W, 3] uint8 (HWC) or [N, 3, H, W] (CHW).
        Returns the same layout as input.
        """
        if video_chunk is None or video_chunk.numel() == 0:
            return video_chunk
        hwc = (video_chunk.shape[-1] == 3)          # True → HWC layout
        if hwc:
            h, w = video_chunk.shape[1], video_chunk.shape[2]
        else:
            h, w = video_chunk.shape[2], video_chunk.shape[3]
        if max(h, w) <= max_side:
            return video_chunk
        scale = max_side / max(h, w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        # Use bilinear interpolation via F.interpolate (CHW float expected)
        if hwc:
            x = video_chunk.float().permute(0, 3, 1, 2)  # → [N,3,H,W]
        else:
            x = video_chunk.float()
        x = torch.nn.functional.interpolate(
            x, size=(new_h, new_w), mode="bilinear", align_corners=False
        ).clamp(0, 255).to(torch.uint8)
        return x.permute(0, 2, 3, 1) if hwc else x

    @torch.inference_mode()
    def encode_video_chunk(self, video_chunk, frame_fps: float = 4.0):
        if video_chunk is None or (hasattr(video_chunk, "shape") and video_chunk.shape[0] == 0):
            return

        # Shared preprocessing: resize + processor call with frame sampling
        # disabled (do_sample_frames=False) so every sampled frame is encoded.
        video_input, grid = preprocess_video_frames(self.processor, video_chunk, frame_fps)
        self.last_video_grid_thw = grid

        pixel_values_videos = video_input["pixel_values_videos"].to(self.device, self.dtype)
        video_grid_thw = video_input["video_grid_thw"].to(self.device)
        video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw)
        if hasattr(video_outputs, "pooler_output") and video_outputs.pooler_output is not None:
            video_features = torch.cat(video_outputs.pooler_output, dim=0).to(self.device, self.dtype).unsqueeze(0)
        else:
            video_features = video_outputs[0].to(self.device, self.dtype).unsqueeze(0)

        self._ensure_dynamic_cache()

        global_offset_per_layer = self._get_next_global_offset_per_layer()
        q_len = video_features.shape[1]
        batch = video_features.shape[0]

        base_offset = global_offset_per_layer[0]
        grid_pos_ids = get_qwen3_vl_position_ids(
            video_grid_thw[0].tolist(),
            q_len,
            offset=base_offset,
            vision_config=self.config.vision_config,
            sample_fps=self.sample_fps,
        ).to(self.device)

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            layer_offset = global_offset_per_layer[layer_idx]
            current_layer_pos = grid_pos_ids.clone()
            if layer_offset != base_offset:
                current_layer_pos = current_layer_pos + (layer_offset - base_offset)
            self._layer_position_ids[layer_idx] = self._build_position_ids_3d_for_vision(
                current_layer_pos, batch
            )

        default_position_ids_3d = self._build_position_ids_3d_for_vision(grid_pos_ids, batch)

        out = self.language_model(
            inputs_embeds=video_features,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
            position_ids=default_position_ids_3d,
        )
        self.kv_cache = out.past_key_values
        contiguous_kv(self.kv_cache)

        for layer_idx in range(self.num_layers):
            layer_offset = global_offset_per_layer[layer_idx]
            current_layer_pos = grid_pos_ids.clone()
            if layer_offset != base_offset:
                current_layer_pos = current_layer_pos + (layer_offset - base_offset)
            self._append_position_ids_layer_explicit(layer_idx, current_layer_pos)

        self.last_encoded_frames = video_chunk.shape[0]
        self.total_processed_frames += video_chunk.shape[0]

        self._layer_position_ids.clear()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # KV cache pruning
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _shrink_positions_and_rerotate_keys(self, keep_indices_per_layer):
        device = self.device
        curr_lens = self._get_cache_seq_len_per_layer()

        for layer_idx in range(self.num_layers):
            layer_len = curr_lens[layer_idx] if layer_idx < len(curr_lens) else curr_lens[0]
            cache = self._position_ids_cache[layer_idx]
            if cache is None or cache.shape[1] != layer_len:
                pos = torch.arange(layer_len, device=device, dtype=torch.float32)
                self._position_ids_cache[layer_idx] = pos.unsqueeze(0).expand(3, -1).clone()

        max_pos_limit = getattr(self.language_model.config, "max_position_embeddings", None)
        if max_pos_limit is None:
            text_cfg = getattr(self.language_model.config, "text_config", None)
            max_pos_limit = getattr(text_cfg, "max_position_embeddings", 262144) if text_cfg else 262144
        compact_threshold = max_pos_limit - 1024

        current_max_pos = 0
        for layer_cache in self._position_ids_cache:
            if layer_cache is not None and layer_cache.numel() > 0:
                current_max_pos = max(current_max_pos, layer_cache.max().item())

        should_compact = (current_max_pos > compact_threshold) if self.streaming else True

        if should_compact:
            logger.info(
                f"[Shrink] Max position {current_max_pos} > {compact_threshold}. "
                "Compacting position IDs (Shift Left)."
            )

        old_position_ids_cache = [
            c.clone() if c is not None else None for c in self._position_ids_cache
        ]

        sample_k = self.kv_cache[0][0]
        dtype = sample_k.dtype
        mrope_section = self._mrope_section
        new_kv_cache = []

        for layer_idx, (k_layer, v_layer) in enumerate(self.kv_cache):
            keep_indices_layer = keep_indices_per_layer[layer_idx]
            if not isinstance(keep_indices_layer, torch.Tensor):
                keep_indices_layer = torch.as_tensor(keep_indices_layer, device=device)

            seq_len_layer = k_layer.shape[2]
            safe_idx = self._sanitize_keep_indices(keep_indices_layer, seq_len_layer)

            if safe_idx.numel() == 0:
                logger.warning(f"Layer {layer_idx}: keep_indices empty; keeping first token")
                safe_idx = torch.tensor([0], device=device)

            is_long_term = (layer_idx >= self.long_term_threshold)

            k_kept = torch.index_select(k_layer, dim=2, index=safe_idx)
            v_kept = torch.index_select(v_layer, dim=2, index=safe_idx)
            old_pos_kept = old_position_ids_cache[layer_idx][:, safe_idx]

            if should_compact:
                new_pos_kept = old_pos_kept.clone()
                if new_pos_kept.shape[1] > 0:
                    text_offset = self.visual_start_idx
                    num_text_kept = (safe_idx < text_offset).sum().item()
                    num_video_kept = (safe_idx >= text_offset).sum().item()

                    if num_video_kept > 0:
                        video_indices_in_kept = torch.arange(
                            num_text_kept, new_pos_kept.shape[1], device=device
                        )
                        old_video_pos = old_pos_kept[:, video_indices_in_kept]
                        for dim in range(3):
                            old_vals = old_video_pos[dim]
                            unique_vals, inverse_indices = torch.unique(
                                old_vals, sorted=True, return_inverse=True
                            )
                            compact_map = torch.arange(len(unique_vals), device=device) + text_offset
                            new_pos_kept[dim, video_indices_in_kept] = compact_map[inverse_indices].to(
                                new_pos_kept.dtype
                            )

                cos_old, sin_old = compute_cos_sin_for_positions(
                    self.language_model, len(safe_idx), old_pos_kept, dtype, device
                )
                cos_new, sin_new = compute_cos_sin_for_positions(
                    self.language_model, len(safe_idx), new_pos_kept, dtype, device
                )
                cos_delta, sin_delta = rotary_delta(cos_old, sin_old, cos_new, sin_new)

                try:
                    k_kept_final = apply_rotary_delta_to_keys_only(
                        k_kept, cos_delta, sin_delta, mrope_section
                    )
                except Exception as e:
                    logger.error(
                        f"apply_rotary_delta failed at layer {layer_idx}: "
                        f"k_kept={tuple(k_kept.shape)}, err={e}"
                    )
                    raise
            else:
                new_pos_kept = old_pos_kept
                k_kept_final = k_kept

            if is_long_term:
                mask = torch.ones(seq_len_layer, dtype=torch.bool, device=device)
                mask[safe_idx] = False
                prune_indices = torch.nonzero(mask).squeeze(1)

                if prune_indices.numel() > 0:
                    k_pruned = torch.index_select(k_layer, dim=2, index=prune_indices)
                    v_pruned = torch.index_select(v_layer, dim=2, index=prune_indices)
                    v_summary = v_pruned.mean(dim=2, keepdim=True)

                    old_pos_pruned = old_position_ids_cache[layer_idx][:, prune_indices]
                    summary_pos_id = new_pos_kept.max().item() + 1
                    summary_pos_tensor = torch.tensor(
                        [summary_pos_id], device=device, dtype=torch.float32
                    ).repeat(3, 1)
                    target_pos_pruned = summary_pos_tensor.expand(3, old_pos_pruned.shape[1])

                    cos_old, sin_old = compute_cos_sin_for_positions(
                        self.language_model, old_pos_pruned.shape[1], old_pos_pruned, dtype, device
                    )
                    cos_new, sin_new = compute_cos_sin_for_positions(
                        self.language_model, target_pos_pruned.shape[1], target_pos_pruned, dtype, device
                    )
                    cos_delta, sin_delta = rotary_delta(cos_old, sin_old, cos_new, sin_new)

                    k_pruned_aligned = apply_rotary_delta_to_keys_only(
                        k_pruned, cos_delta, sin_delta, mrope_section
                    )
                    k_summary_final = k_pruned_aligned.mean(dim=2, keepdim=True)

                    k_final = torch.cat([k_kept_final, k_summary_final], dim=2)
                    v_final = torch.cat([v_kept, v_summary], dim=2)
                    new_pos_layer = torch.cat([new_pos_kept, summary_pos_tensor], dim=1)
                else:
                    k_final = k_kept_final
                    v_final = v_kept
                    new_pos_layer = new_pos_kept
            else:
                k_final = k_kept_final
                v_final = v_kept
                new_pos_layer = new_pos_kept

            new_kv_cache.append((k_final.contiguous(), v_final.contiguous()))
            self._position_ids_cache[layer_idx] = new_pos_layer.clone()

        self.kv_cache = DynamicCache.from_legacy_cache(new_kv_cache)
        contiguous_kv(self.kv_cache)

        new_lens = self._get_cache_seq_len_per_layer()
        logger.info(
            f"Layer-wise shrink done. Lengths: min={min(new_lens)}, max={max(new_lens)}, "
            f"first={new_lens[0]}, last={new_lens[-1]}"
        )

    @torch.inference_mode()
    def apply_kv_cache_pruning_strict(self, keep_indices_all_layers):
        if self.kv_cache is None:
            logger.warning("No KV-Cache to prune")
            return
        if not keep_indices_all_layers or len(keep_indices_all_layers[0]) == 0:
            logger.warning("Empty keep_indices; skip pruning")
            return
        self._shrink_positions_and_rerotate_keys(keep_indices_all_layers)
        logger.info(f"Strict-shrunk KV cache. New length: {get_cache_seq_len(self.kv_cache)}")

    def allocate_budget_by_depth(self, total_budget, num_layers):
        budget_per_layer = [total_budget // num_layers] * num_layers
        diff = total_budget - sum(budget_per_layer)
        budget_per_layer[-1] += diff
        return budget_per_layer

    # ------------------------------------------------------------------
    # Attention-based pruning decision
    # ------------------------------------------------------------------

    def _compute_attention_scores_manually(self, input_ids, past_key_values):
        """
        Compute per-layer attention weights without flash-attention
        (compatible with 4-bit quantized models).
        """
        device = self.device
        global_offset_per_layer = self._get_next_global_offset_per_layer()
        q_len = input_ids.shape[1]
        batch = input_ids.shape[0]

        inputs_embeds = self.get_input_embeddings()(input_ids)

        lm_config = self.language_model.config
        text_cfg = getattr(lm_config, "text_config", lm_config)
        num_layers = text_cfg.num_hidden_layers
        num_heads = text_cfg.num_attention_heads
        num_key_value_heads = text_cfg.num_key_value_heads
        head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // num_heads)
        hidden_size = text_cfg.hidden_size

        attention_weights_list = []
        hidden_states = inputs_embeds

        for layer_idx in range(num_layers):
            layer = self.language_model.layers[layer_idx]
            past_k, past_v = past_key_values[layer_idx]

            layer_offset = global_offset_per_layer[layer_idx]
            position_ids_3d = torch.zeros((3, 1, q_len), device=device, dtype=torch.float32)
            for dim in range(3):
                position_ids_3d[dim, 0, :] = torch.arange(
                    layer_offset, layer_offset + q_len, device=device
                )

            hidden_states_norm = layer.input_layernorm(hidden_states)
            attn = layer.self_attn

            query_states = attn.q_proj(hidden_states_norm)
            query_states = query_states.view(batch, q_len, num_heads, head_dim).transpose(1, 2)

            key_states = attn.k_proj(hidden_states_norm)
            key_states = key_states.view(batch, q_len, num_key_value_heads, head_dim).transpose(1, 2)

            value_states = attn.v_proj(hidden_states_norm)
            value_states = value_states.view(batch, q_len, num_key_value_heads, head_dim).transpose(1, 2)

            rotary_emb = _get_rotary_module(self.language_model)
            dummy_h = torch.zeros((1, q_len, hidden_size), device=device, dtype=hidden_states.dtype)
            cos, sin = rotary_emb(dummy_h, position_ids_3d)

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)

            if num_key_value_heads != num_heads:
                n_rep = num_heads // num_key_value_heads
                key_states = torch.repeat_interleave(key_states, n_rep, dim=1)
                value_states = torch.repeat_interleave(value_states, n_rep, dim=1)

            attn_weights = torch.matmul(
                query_states.float(), key_states.float().transpose(-2, -1)
            ) / (head_dim ** 0.5)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float16).to(
                query_states.dtype
            )
            attention_weights_list.append(attn_weights)

        return attention_weights_list

    def prune_kv_cache_by_attention(
        self, attn_weights_local, attn_weights_global, attn_weights_mixed, num_keep=2000
    ):
        device = self.device
        visual_start_idx = self.visual_start_idx
        num_layers = len(attn_weights_local)

        question_len_local = attn_weights_local[0].shape[2]
        question_len_global = attn_weights_global[0].shape[2]
        question_len_mixed = attn_weights_mixed[0].shape[2]

        total_budget = num_keep * num_layers
        budget_per_layer = self.allocate_budget_by_depth(total_budget, num_layers)

        layer_raw_scores = []
        layer_configs = []

        for layer_idx in range(num_layers):
            if layer_idx < self.short_term_threshold:
                layer_type = "short-term"
                layer_attn_weights = attn_weights_local[layer_idx]
                question_len = question_len_local
                layer_recency_alpha = 1
                k = 20
            elif layer_idx >= self.long_term_threshold:
                layer_type = "long-term"
                layer_attn_weights = attn_weights_global[layer_idx]
                question_len = question_len_global
                layer_recency_alpha = 0
                k = 0.0
            else:
                layer_type = "mid-term"
                layer_attn_weights = attn_weights_mixed[layer_idx]
                question_len = question_len_mixed
                progress = (layer_idx - self.short_term_threshold) / (
                    self.long_term_threshold - self.short_term_threshold
                )
                layer_recency_alpha = 0.75 - 0.6 * progress
                k = 20 - 12 * progress

            visual_attn_weights = (
                layer_attn_weights[0].mean(dim=0)[:, visual_start_idx: -1 * question_len].mean(dim=0)
            )
            num_visual_tokens = visual_attn_weights.shape[0]
            layer_budget = budget_per_layer[layer_idx]

            if layer_type == "long-term":
                layer_budget = max(0, layer_budget - 1)

            positions = torch.arange(num_visual_tokens, device=device, dtype=torch.float32)
            time_distances = (num_visual_tokens - 1 - positions) / max(num_visual_tokens - 1, 1)
            recency_weights = torch.exp(-k * time_distances)

            attn_norm = (visual_attn_weights - visual_attn_weights.min()) / (
                visual_attn_weights.max() - visual_attn_weights.min() + 1e-6
            )
            recency_norm = (recency_weights - recency_weights.min()) / (
                recency_weights.max() - recency_weights.min() + 1e-6
            )
            raw_score = attn_norm * (1 - layer_recency_alpha) + recency_norm * layer_recency_alpha

            layer_raw_scores.append(raw_score)
            layer_configs.append({
                "budget": min(layer_budget, num_visual_tokens),
                "layer_type": layer_type,
                "visual_start_idx": visual_start_idx,
            })

        # Inter-layer consistency refinement
        refined_scores = [s.clone() for s in layer_raw_scores]
        for i in range(len(refined_scores) - 2, -1, -1):
            current_type = layer_configs[i]["layer_type"]
            gamma = {"long-term": 0.4, "mid-term": 0.3, "short-term": 0.1}[current_type]

            score_current = refined_scores[i]
            score_next = refined_scores[i + 1]

            if score_current.shape[0] != score_next.shape[0]:
                score_next = F.interpolate(
                    score_next.view(1, 1, -1),
                    size=score_current.shape[0],
                    mode="linear",
                    align_corners=False,
                ).view(-1)
            refined_scores[i] = (1 - gamma) * score_current + gamma * score_next

        keep_indices_all_layers = []
        for layer_idx, score in enumerate(refined_scores):
            cfg = layer_configs[layer_idx]
            actual_num_keep = cfg["budget"]
            start_idx = cfg["visual_start_idx"]

            topk_indices_relative = torch.topk(score, actual_num_keep, sorted=False)[1]
            topk_indices_absolute = topk_indices_relative + start_idx
            topk_indices_absolute_sorted = torch.sort(topk_indices_absolute)[0]

            keep_indices = torch.cat([
                torch.arange(start_idx, device=device),
                topk_indices_absolute_sorted,
            ]).tolist()
            keep_indices_all_layers.append(keep_indices)

        return keep_indices_all_layers

    # ------------------------------------------------------------------
    # Pseudo-forward with safety-specific questions
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_next_question(self):
        """
        Generate safety-specific local and global questions for KV pruning.
        These guide the attention-based pruning to retain safety-relevant tokens.
        """
        if hasattr(self, "conv_history") and len(self.conv_history) > 0:
            last_q, last_a, last_options = self.conv_history[-1]

            # Expand option letter to full text
            option_match = re.match(r"^\s*(?:\()?([A-Z])(?:\))?.?\s*$", last_a)
            if option_match and last_options is not None:
                option_char = option_match.group(1)
                option_idx = ord(option_char) - ord("A")
                if 0 <= option_idx < len(last_options):
                    last_a = last_options[option_idx]
            last_a = re.sub(r"[A-Z]\) ", "", last_a)

            context = f"Context: Question: {last_q} Answer: {last_a}."
            local_query = (
                f"{context} Describe the current scene focusing on workers, equipment "
                "positions, PPE compliance, proximity to hazards, and any unsafe actions."
            )
            global_query = (
                f"{context} Summarize the overall safety situation, noting recurring "
                "violations, escalating risk patterns, and unresolved hazards."
            )
        else:
            local_query = (
                "Describe the current scene focusing on workers, equipment positions, "
                "PPE compliance, proximity to hazards, and any unsafe actions."
            )
            global_query = (
                "Summarize the overall safety situation, noting recurring violations, "
                "escalating risk patterns, and unresolved hazards."
            )

        return local_query, global_query

    @torch.inference_mode()
    def pseudo_forward(self, local_question=None, global_question=None):
        device = self.device

        if local_question is None:
            local_question = (
                "Describe the current scene focusing on workers and safety hazards."
            )
        if global_question is None:
            global_question = "Summarize the overall safety situation in the video."

        # Always use manual attention scoring (compatible with 4-bit quantized models)
        local_input_ids = self.processor.tokenizer(local_question).input_ids
        local_input_ids = torch.as_tensor([local_input_ids], device=device, dtype=torch.int)

        global_offset_per_layer = self._get_next_global_offset_per_layer()
        q_len_local = local_input_ids.shape[1]
        batch = local_input_ids.shape[0]

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            position_ids_3d = self._build_position_ids_3d_for_text(
                global_offset_per_layer[layer_idx], q_len_local, batch
            )
            self._layer_position_ids[layer_idx] = position_ids_3d

        attn_weights_local = self._compute_attention_scores_manually(local_input_ids, self.kv_cache)

        global_input_ids = self.processor.tokenizer(global_question).input_ids
        global_input_ids = torch.as_tensor([global_input_ids], device=device, dtype=torch.int)
        q_len_global = global_input_ids.shape[1]

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            self._layer_position_ids[layer_idx] = self._build_position_ids_3d_for_text(
                global_offset_per_layer[layer_idx], q_len_global, batch
            )

        attn_weights_global = self._compute_attention_scores_manually(global_input_ids, self.kv_cache)

        mixed_question = local_question + "; " + global_question
        mixed_input_ids = self.processor.tokenizer(mixed_question).input_ids
        mixed_input_ids = torch.as_tensor([mixed_input_ids], device=device, dtype=torch.int)
        q_len_mixed = mixed_input_ids.shape[1]

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            self._layer_position_ids[layer_idx] = self._build_position_ids_3d_for_text(
                global_offset_per_layer[layer_idx], q_len_mixed, batch
            )

        attn_weights_mixed = self._compute_attention_scores_manually(mixed_input_ids, self.kv_cache)

        self._layer_position_ids.clear()

        print(f"GPU memory usage: {self.get_gpu_memory_usage_gb():.2f} GB")
        current_k_states_len = self.kv_cache[0][0].shape[2]

        keep_indices_all_layers = self.prune_kv_cache_by_attention(
            attn_weights_local, attn_weights_global, attn_weights_mixed,
            num_keep=self.kv_size,
        )

        if current_k_states_len > self.kv_size:
            print(f"Applying KV-Cache compression (k_states={current_k_states_len} > kv_size={self.kv_size})")
            self.apply_kv_cache_pruning_strict(keep_indices_all_layers)

    @torch.inference_mode()
    def predict_and_compress(self):
        local_question, global_question = self.predict_next_question()
        self.pseudo_forward(local_question, global_question)

    # ------------------------------------------------------------------
    # Question answering (generation)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def question_answering(
        self,
        input_text,
        max_new_tokens=128,
        temperature=0,
        repetition_penalty=1.1,
    ):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]
        output_ids = []

        start_time = time.perf_counter()

        prompt = input_text["prompt"]
        input_ids = self.processor.tokenizer(prompt).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)

        self._ensure_dynamic_cache()
        past_lens_prefill = self._get_cache_seq_len_per_layer()
        global_offset_prefill = self._get_next_global_offset_per_layer()

        inputs_embeds = self.get_input_embeddings()(input_ids)
        q_len_prefill = inputs_embeds.shape[1]
        batch = inputs_embeds.shape[0]

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            self._layer_position_ids[layer_idx] = self._build_position_ids_3d_for_text(
                global_offset_prefill[layer_idx], q_len_prefill, batch
            )

        position_ids_3d = self._build_position_ids_3d_for_text(
            global_offset_prefill[0], q_len_prefill, batch
        )

        out = self.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=self.kv_cache,
            position_ids=position_ids_3d,
        )
        past_key_values = out.past_key_values
        logits = self.lm_head(out.last_hidden_state)

        for layer_idx in range(self.num_layers):
            offset = global_offset_prefill[layer_idx]
            self._append_position_ids_layer(layer_idx, [offset, offset, offset], q_len_prefill)
        self._layer_position_ids.clear()

        for step in range(max_new_tokens):
            last_token_logits = logits[0, -1, :]

            if repetition_penalty != 1.0 and output_ids:
                for token_id in set(output_ids):
                    if last_token_logits[token_id] < 0:
                        last_token_logits[token_id] *= repetition_penalty
                    else:
                        last_token_logits[token_id] /= repetition_penalty

            if temperature == 0.0:
                _, indices = torch.topk(last_token_logits, 1)
                token = int(indices[0])
            else:
                scaled_logits = last_token_logits / temperature
                scaled_logits = torch.nan_to_num(
                    scaled_logits, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")
                )
                probs = F.softmax(scaled_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs = probs / probs_sum
                    token = torch.multinomial(probs, num_samples=1).item()
                else:
                    _, indices = torch.topk(last_token_logits, 1)
                    token = int(indices[0])

            output_ids.append(token)

            if step == 0:
                end_time = time.perf_counter()
                print(f"TTFT: {end_time - start_time:.3f}s")

            if token in stop_token_ids:
                break

            curr_global_offset = self._get_next_global_offset_per_layer()
            self._layer_position_ids.clear()
            for layer_idx in range(self.num_layers):
                self._layer_position_ids[layer_idx] = self._build_position_ids_3d_for_text(
                    curr_global_offset[layer_idx], 1, 1
                )

            position_ids_3d = self._build_position_ids_3d_for_text(curr_global_offset[0], 1, 1)

            out = self.language_model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
                position_ids=position_ids_3d,
            )
            logits = self.lm_head(out.last_hidden_state)
            past_key_values = out.past_key_values

            for layer_idx in range(self.num_layers):
                offset = curr_global_offset[layer_idx]
                self._append_position_ids_layer(layer_idx, [offset, offset, offset], 1)
            self._layer_position_ids.clear()

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        # Save to conversation history
        current_question = input_text.get("question", "")
        formatted_question = input_text.get("formatted_question", None)
        current_options = None
        if formatted_question:
            option_matches = re.findall(
                r"\([A-Z]\)\s*(.+?)(?=\n\([A-Z]\)|\nThe best answer|\n*$)",
                formatted_question,
                re.DOTALL,
            )
            if option_matches:
                current_options = [opt.strip() for opt in option_matches]
        self.conv_history.append((current_question, output, current_options))

        # Restore KV cache to pre-QA state (truncate QA tokens)
        self.kv_cache = past_key_values
        self._truncate_kv_cache(past_lens_prefill)
        for layer_idx in range(self.num_layers):
            cache = self._position_ids_cache[layer_idx]
            if cache is not None and cache.shape[1] > past_lens_prefill[layer_idx]:
                self._position_ids_cache[layer_idx] = cache[
                    :, : past_lens_prefill[layer_idx]
                ].contiguous()

        new_lens = self._get_cache_seq_len_per_layer()
        print(f"Cache after QA: min={min(new_lens)}, max={max(new_lens)}")
        torch.cuda.empty_cache()
        return output


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def load_hermes_model(
    model_path: str = "Qwen/Qwen3-VL-4B-Instruct",
    kv_size: int = 2000,
    streaming: bool = True,
    sample_fps: float = 0.5,
    load_in_4bit: bool = True,
    device: str = "cuda",
) -> tuple:
    """
    Load Qwen3-VL-4B with HERMES KV-cache management.

    Args:
        model_path: HuggingFace model path or local directory.
        kv_size: Maximum number of visual KV tokens to retain per layer.
                 Recommended 2000 for 6 GB VRAM GPUs.
        streaming: If True, position ID compaction only triggers near context limit.
                   If False, compaction runs after every chunk (offline mode).
        sample_fps: Frame sampling rate used for temporal position IDs.
        load_in_4bit: Use 4-bit BitsAndBytes quantization (required for 6 GB VRAM).
        device: Target device for init prompt tensor.

    Returns:
        (model, processor) tuple.
    """
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    # Cap resolution to prevent OOM on high-res input (256x256 px budget = 65536 pixels).
    # Qwen3-VL processors expose min_pixels / max_pixels on the image processor.
    _ip = getattr(processor, "image_processor", None)
    if _ip is not None:
        _ip.min_pixels = 4 * 28 * 28      # 3136 — absolute minimum
        _ip.max_pixels = 256 * 28 * 28    # 200704 — ~448×448 effective cap
    del _ip

    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    init_prompt_ids = processor.tokenizer(system_prompt, return_tensors="pt").input_ids.to(device)

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"],
            llm_int8_enable_fp32_cpu_offload=True,
        )
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    else:
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
        )

    # Construct HERMES wrapper via __new__ + dict copy (HERMES pattern)
    model = Qwen3VLHermes.__new__(Qwen3VLHermes)
    model.__dict__ = base_model.__dict__.copy()

    AbstractHermes.__init__(model, processor, init_prompt_ids.tolist(), kv_size)
    model.streaming = streaming
    model.sample_fps = sample_fps

    # Qwen3-VL-4B: num_hidden_layers = 36
    text_cfg = getattr(base_model.config, "text_config", base_model.config)
    num_layers = text_cfg.num_hidden_layers
    model.num_layers = num_layers
    model._position_ids_cache = [None for _ in range(num_layers)]

    model.short_term_ratio = 0.1
    model.long_term_ratio = 0.3
    model.short_term_threshold = int(num_layers * model.short_term_ratio)   # 3
    model.long_term_threshold = int(num_layers * (1 - model.long_term_ratio))  # 25
    model.total_processed_frames = 0

    model._mrope_section = _get_mrope_section(base_model)  # (24, 20, 20)
    model._layer_position_ids = {}
    model._hook_handles = []
    model._register_forward_hooks()

    logger.info(f"Loaded Qwen3VLHermes: layers={num_layers}, kv_size={kv_size}, "
                f"mrope_section={model._mrope_section}")

    model.eval()
    return model, processor
