"""
Shared Qwen3-VL video preprocessing used by all streaming runners (HERMES and
InfiniPot-V) so that identical frames produce identical model inputs.

Why this exists (benchmark-fairness fix, 2026-10):
  1. The Qwen3-VL video processor silently re-samples input frames at 2 fps
     (minimum 4 frames) whenever `video_metadata` is missing, collapsing any
     chunk to ~4 frames regardless of how many were sent. We disable sampling
     (the chunker already sampled) and declare the true frame rate.
  2. HERMES passed pre-resized tensors while the InfiniPot runner routed PIL
     images through qwen_vl_utils, which applied a different resize policy.
     Both now go through this single code path.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers.video_utils import VideoMetadata

MAX_SIDE = 448  # matches the original hermes_engine._resize_frames


def frames_to_tensor(frames) -> torch.Tensor:
    """[N,H,W,3] uint8 (numpy/torch, HWC or CHW) -> [N,3,H,W] uint8."""
    if isinstance(frames, torch.Tensor):
        t = frames
    else:
        t = torch.from_numpy(np.stack([np.asarray(f) for f in frames], axis=0))
    if t.ndim != 4:
        raise ValueError(f"Expected 4-D frame tensor, got {tuple(t.shape)}")
    if t.shape[-1] == 3:  # HWC -> CHW
        t = t.permute(0, 3, 1, 2)
    return t.contiguous()


def resize_frames(video_tensor: torch.Tensor, max_side: int = MAX_SIDE) -> torch.Tensor:
    """Clamp the longest side to `max_side` (bilinear), bounding vision tokens."""
    n, c, h, w = video_tensor.shape
    if max(h, w) <= max_side:
        return video_tensor
    scale = max_side / float(max(h, w))
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    x = torch.nn.functional.interpolate(
        video_tensor.float(), size=(new_h, new_w), mode="bilinear", align_corners=False
    )
    return x.clamp(0, 255).to(torch.uint8)


def preprocess_video_frames(processor, frames, fps: float):
    """
    Runs the Qwen3-VL processor on pre-sampled frames WITHOUT re-sampling.

    Returns (processor_outputs, grid) where grid = [T, H, W] patch grid with
    T == frames//2 (temporal_patch_size=2 merges frame pairs). The caller must
    move tensors to the target device.
    """
    video_tensor = resize_frames(frames_to_tensor(frames))
    n, _, h, w = video_tensor.shape
    if n % 2 == 1:  # temporal_patch_size=2 requires an even frame count
        video_tensor = torch.cat([video_tensor, video_tensor[-1:]], dim=0)
        n += 1
    metadata = VideoMetadata(
        total_num_frames=n,
        fps=float(fps),
        height=int(h),
        width=int(w),
        frames_indices=list(range(n)),  # we send every sampled frame, in order
    )
    out = processor(
        text=[""],
        videos=[video_tensor],
        video_metadata=[metadata],
        do_sample_frames=False,
        return_tensors="pt",
    )
    grid = out["video_grid_thw"][0].tolist()
    if grid[0] * 2 != n:
        raise RuntimeError(f"Frame handling mismatch: sent {n} frames, processor grid {grid}")
    return out, grid


def video_position_ids_3d(
    grid,
    offset: float,
    vision_config=None,
    sample_fps: float = 0.5,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """
    HERMES-style 3D M-RoPE positions for one video block.

    grid:   [T, H, W] patch grid from the processor (pre-spatial-merge).
    offset: first free position in each RoPE dimension.
    Returns a [3, seq] float tensor in post-merge token order (t-major,
    then row, then column), matching get_video_features output order.
    """
    t, h, w = grid
    sm = getattr(vision_config, "spatial_merge_size", spatial_merge_size) if vision_config else spatial_merge_size
    lt, lh, lw = t, h // sm, w // sm

    tps = getattr(vision_config, "tokens_per_second", None) if vision_config else None
    if tps:
        # 2 frames are merged per temporal slot; sample_fps is the source frame rate
        second_per_grid_t = 2.0 / sample_fps
        tt = (
            torch.arange(lt, dtype=torch.float32).view(-1, 1) * second_per_grid_t * tps
        ).expand(-1, lh * lw).flatten() + offset
    else:
        tt = torch.arange(lt, dtype=torch.float32).repeat_interleave(lh * lw) + offset
    hh = torch.arange(lh, dtype=torch.float32).view(1, -1, 1).expand(lt, -1, lw).flatten() + offset
    ww = torch.arange(lw, dtype=torch.float32).view(1, 1, -1).expand(lt, lh, -1).flatten() + offset
    return torch.stack([tt, hh, ww])  # [3, seq]
