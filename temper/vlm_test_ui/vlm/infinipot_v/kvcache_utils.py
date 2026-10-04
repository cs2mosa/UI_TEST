"""
InfiniPot-V Key-Value Cache Compression Utility.

Ported from aiha-lab/InfiniPot-V reference implementation and adapted for Qwen3-VL
architectures (verified against Qwen/Qwen3-VL-4B-Instruct in 4-bit BitsAndBytes).
See docs/qwen3vl_internals.md for internal architecture verification.
"""

from __future__ import annotations
from typing import Tuple, Any
import torch


def process_kv_cache(
    past_key_values: Any,
    model: Any,
    system_size: int,
    inst_size: int,
    token_per_frame: int,
    compress_frame_num: int,
    method: str = "uniform",
    tar_ratio: float = 0.5,
    query_ratio: float = 0.25,
    adaptive_pooling: bool = False,
    is_first_block: bool = False,
    is_last_block: bool = False,
    per_frame: bool = False,
) -> Tuple[Any, Any]:
    """
    Compress KV cache by keeping only a specified number of frames in vision tokens.

    Args:
        past_key_values: Current KV cache (DynamicCache with .layers[i].keys/values).
        model: The model instance (or mock containing model.model.language_model.layers).
        system_size: Number of prefix system tokens (never compressed or modified).
        inst_size: Size of instruction tokens.
        token_per_frame: Number of visual tokens per frame.
        compress_frame_num: Target number of frames to retain in cache.
        method: Compression strategy ('swa', 'uniform', 'infinipot-v' / 'tar_val').
        tar_ratio: Ratio for TaR (Temporal-aware Retrieval) vs VaN in 'infinipot-v'.
        query_ratio: Ratio of query frames used for TaR cosine similarity.
        adaptive_pooling: Whether to apply layer-adaptive pooling to value norm scores.
        is_first_block: Whether this is the first block in the sequence.
        is_last_block: Whether this is the final block in the sequence.
        per_frame: Whether to select complete frames (reserved for future extensions).

    Returns:
        tuple: (compressed_past_key_values, cap_list)
    """
    pooling_func_list = [3, 2, 1, -1]
    pool_size_list = [7, 5, 3, 1]

    if compress_frame_num <= 0:
        return past_key_values, None

    # Determine sequence length
    if hasattr(past_key_values, "layers") and len(past_key_values.layers) > 0 and past_key_values.layers[0].keys is not None:
        current_seq_len = past_key_values.layers[0].keys.shape[2]
    elif hasattr(past_key_values, "get_seq_length"):
        current_seq_len = past_key_values.get_seq_length()
    else:
        current_seq_len = past_key_values[0][0].shape[2]

    # Calculate vision token range
    vision_start = system_size
    vision_end = current_seq_len
    vision_length = vision_end - vision_start
    current_frame_num = vision_length // token_per_frame
    # Clamp to complete frames: dynamic-resolution encoders may leave a fractional tail.
    # Floor to the nearest multiple rather than asserting, so callers never crash.
    vision_length = current_frame_num * token_per_frame

    if current_frame_num <= compress_frame_num:
        return past_key_values, None

    # Determine number of layers
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        num_layers = len(model.model.language_model.layers)
    elif hasattr(past_key_values, "layers"):
        num_layers = len(past_key_values.layers)
    elif hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        num_layers = len(past_key_values)

    # Process each layer
    for layer_idx in range(num_layers):
        # Layer-Adaptive Pooling configuration
        if adaptive_pooling:
            idx = layer_idx // max(1, (num_layers // len(pooling_func_list)))
            idx = min(idx, len(pooling_func_list) - 1)
            avg_pooling_nd = pooling_func_list[idx]
            attn_pool_size = pool_size_list[idx]
        else:
            avg_pooling_nd = -1
            attn_pool_size = 1

        if hasattr(past_key_values, "layers"):
            key_states = past_key_values.layers[layer_idx].keys
            value_states = past_key_values.layers[layer_idx].values
        else:
            key_states, value_states = past_key_values[layer_idx]

        bsz, num_heads, q_len, head_dim = key_states.shape

        # Extract vision tokens to compress (preserving system tokens untouched)
        key_states_to_compress = key_states[:, :, system_size:, :]
        value_states_to_compress = value_states[:, :, system_size:, :]

        # Calculate total tokens to keep
        total_tokens_to_keep = compress_frame_num * token_per_frame

        if method == "swa":
            # Sliding Window Attention (SWA): Keep the most recent visual tokens
            start_idx = key_states_to_compress.shape[2] - total_tokens_to_keep
            all_indices = torch.arange(start_idx, key_states_to_compress.shape[2], device=key_states.device)
            all_indices = all_indices.unsqueeze(0).expand(num_heads, -1)

        elif method == "uniform":
            # Uniform: Sample tokens uniformly across the visual sequence
            total_tokens = key_states_to_compress.shape[2]
            step = total_tokens / total_tokens_to_keep
            selected_token_indices = [int(i * step) for i in range(total_tokens_to_keep)]
            all_indices = torch.tensor(selected_token_indices, device=key_states.device, dtype=torch.long)
            all_indices = all_indices.unsqueeze(0).expand(num_heads, -1)

        elif method in ("infinipot-v", "tar_val"):
            # Combined TaR (Temporal-aware Retrieval) + VaN (Value Norm) approach
            attn_budget = round((1 - tar_ratio) * compress_frame_num * token_per_frame)

            # TaR part
            if tar_ratio > 0:
                tar_budget = compress_frame_num * token_per_frame - attn_budget
                query_frame = max(1, int(current_frame_num * query_ratio))
                query_length = int(query_frame * token_per_frame)
                total_length = key_states_to_compress.shape[2]

                # Get query and key embeddings
                query_frame_emb = key_states_to_compress[:, :, -query_length:, :]
                query_emb_norm = query_frame_emb / (query_frame_emb.norm(dim=-1, keepdim=True) + 1e-9)

                key_frame_emb = key_states_to_compress[:, :, :-query_length, :]
                key_emb_norm = key_frame_emb / (key_frame_emb.norm(dim=-1, keepdim=True) + 1e-9)

                # Pix2pix cosine similarity calculation
                query_emb_norm_reshaped = query_emb_norm.reshape(
                    query_emb_norm.shape[0], query_emb_norm.shape[1],
                    query_frame, token_per_frame, query_emb_norm.shape[-1]
                )
                key_emb_norm_reshaped = key_emb_norm.reshape(
                    key_emb_norm.shape[0], key_emb_norm.shape[1],
                    -1, token_per_frame, key_emb_norm.shape[-1]
                )

                key_score = -(
                    query_emb_norm_reshaped.unsqueeze(3) *
                    key_emb_norm_reshaped.unsqueeze(2)
                ).sum(dim=-1).mean(dim=2).reshape(query_emb_norm.shape[0], query_emb_norm.shape[1], -1)

                recent_indices = torch.arange(
                    total_length - query_length, total_length, device=key_states_to_compress.device
                ).unsqueeze(0).expand(num_heads, -1)

                tar_needed = tar_budget - query_length
                if tar_needed > 0:
                    selected_indices = key_score.topk(tar_needed, dim=-1).indices.squeeze(0)
                    tar_indices_combined = torch.cat([selected_indices, recent_indices], dim=-1)
                else:
                    tar_indices_combined = recent_indices
                tar_indices, _ = tar_indices_combined.sort(dim=-1)
            else:
                tar_indices = None

            # Value norm part
            val_norm_score = value_states_to_compress.norm(dim=-1)[0, :, :]  # [num_heads, seq_len]

            # Apply pooling if enabled
            if avg_pooling_nd > 0:
                pool_size = attn_pool_size
                head_num, _ = val_norm_score.shape
                frame_num = val_norm_score.shape[-1] // token_per_frame

                if avg_pooling_nd == 1:
                    avg_pool = torch.nn.AvgPool1d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
                elif avg_pooling_nd == 2:
                    avg_pool = torch.nn.AvgPool2d(kernel_size=(pool_size, pool_size), stride=(1, 1), padding=(pool_size // 2, pool_size // 2))
                elif avg_pooling_nd == 3:
                    avg_pool = torch.nn.AvgPool3d(kernel_size=(pool_size, pool_size, pool_size), stride=(1, 1, 1), padding=(pool_size // 2, pool_size // 2, pool_size // 2))
                else:
                    raise NotImplementedError("We support 1D - 3D Pooling")

                patch_size = getattr(getattr(model, "config", None), "height_width", None)
                if patch_size is None or patch_size <= 0:
                    patch_size = int(token_per_frame ** 0.5) if int(token_per_frame ** 0.5) ** 2 == token_per_frame else 14

                if avg_pooling_nd > 1 and patch_size > 0 and token_per_frame % patch_size == 0:
                    original_numel = val_norm_score.numel()
                    val_norm_reshaped = val_norm_score.reshape(head_num, frame_num, patch_size, -1)

                    T = val_norm_reshaped.shape[-3]
                    H = val_norm_reshaped.shape[-2]
                    W = val_norm_reshaped.shape[-1]

                    pool_size_eff = min(pool_size, T, H, W)
                    if pool_size_eff % 2 == 0:
                        pool_size_eff -= 1

                    if pool_size_eff >= 2:
                        if pool_size_eff != pool_size:
                            pool_size = pool_size_eff
                            if avg_pooling_nd == 2:
                                avg_pool = torch.nn.AvgPool2d(
                                    kernel_size=(pool_size, pool_size),
                                    stride=(1, 1),
                                    padding=(pool_size // 2, pool_size // 2),
                                )
                            elif avg_pooling_nd == 3:
                                avg_pool = torch.nn.AvgPool3d(
                                    kernel_size=(pool_size, pool_size, pool_size),
                                    stride=(1, 1, 1),
                                    padding=(pool_size // 2, pool_size // 2, pool_size // 2),
                                )
                        val_norm_pooled = avg_pool(val_norm_reshaped.float()).to(val_norm_reshaped.dtype)
                        val_norm_score = val_norm_pooled.reshape(head_num, -1)
                        assert original_numel == val_norm_score.numel()
                else:
                    # Fallback to 1D pooling across sequence
                    avg_pool_1d = torch.nn.AvgPool1d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
                    val_norm_score = avg_pool_1d(val_norm_score.unsqueeze(0).float()).squeeze(0).to(val_norm_score.dtype)

            # Combine TaR and Value norm scoring
            if tar_indices is not None:
                head_indices = torch.arange(num_heads, device=key_states.device).unsqueeze(1).expand(-1, tar_indices.size(1))
                val_norm_score[head_indices, tar_indices] = val_norm_score.max() + 1.0

            # Final selection based on combined scores
            all_indices = val_norm_score.topk(compress_frame_num * token_per_frame, dim=-1).indices
            all_indices, _ = all_indices.sort(dim=-1)

        else:
            raise ValueError(
                f"Unknown compression method: {method}. Available methods: 'swa', 'uniform', 'infinipot-v', 'tar_val'"
            )

        # Apply compression using advanced indexing
        batch_indices = torch.zeros_like(all_indices)
        head_indices = torch.arange(num_heads, device=key_states.device).unsqueeze(1).expand(-1, all_indices.size(1))

        assert all_indices.max() < key_states_to_compress.shape[2], (
            f"Selected index max {all_indices.max()} exceeds key length {key_states_to_compress.shape[2]}"
        )

        # Gather selected visual tokens
        new_key_states_to_compress = key_states_to_compress[batch_indices, head_indices, all_indices].unsqueeze(0)
        new_value_states_to_compress = value_states_to_compress[batch_indices, head_indices, all_indices].unsqueeze(0)

        # Re-attach system prefix tokens unchanged
        new_k = torch.cat([key_states[:, :, :system_size, :], new_key_states_to_compress], dim=2)
        new_v = torch.cat([value_states[:, :, :system_size, :], new_value_states_to_compress], dim=2)

        if hasattr(past_key_values, "layers"):
            past_key_values.layers[layer_idx].keys = new_k
            past_key_values.layers[layer_idx].values = new_v
        else:
            past_key_values[layer_idx] = (new_k, new_v)

    if hasattr(past_key_values, "_seen_tokens"):
        past_key_values._seen_tokens = new_k.shape[2]

    return past_key_values, None
