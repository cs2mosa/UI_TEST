from __future__ import annotations
"""
jev_adapter.py - JEV decider interface, validate_probs, evaluate_tick, OneJevDecider.

Imports: dataclasses, typing, time, math, cv2, numpy, app.core.*.
No torch/transformers at module level (lazy, I-5).
"""
import math
import os
import time
from typing import Protocol, runtime_checkable

import numpy as np

from app.core.jev_gate import GateConfig, HAZARDS
from app.core.frame_sampler import (
    sample_timestamps,
    read_frames_at,
    resize_max_side,
    make_thumb,
)
from app.core.types import TickRecord

# ---------------------------------------------------------------------------
# Verbatim from hazard_monitor.HAZARDS (D-3, D-6, rule 3 in §0.2)
# ---------------------------------------------------------------------------
HAZARD_QUESTIONS: dict[str, str] = {
    "pedestrian_vehicle": (
        "A person on foot is in the path of, or very close to, a moving forklift, "
        "pallet jack or other powered vehicle, so that a collision or near-miss could occur "
        "(a person merely visible in the same aisle at a safe distance does not count)"
    ),
    "no_ppe": (
        "At least one person in an active work area is not wearing a high-visibility vest "
        "or a hard hat (a person wearing both does not count)"
    ),
    "blocked_exit": (
        "A fire exit, emergency door, fire extinguisher, emergency equipment or a marked "
        "aisle is obstructed by pallets, boxes, equipment or other objects"
    ),
    "unsafe_load": (
        "A load is unsafe: stacked goods are leaning, collapsing or overhanging, a vehicle is "
        "moving with its load raised high or tilted, or items are falling or about to fall"
    ),
    "fall_or_spill": (
        "A person is on the floor after a fall or slip, or there is a liquid spill, debris "
        "or an object on the walking or driving surface that creates a slip, trip or collision risk"
    ),
}

# ---------------------------------------------------------------------------
# JevDecider Protocol (§7.3)
# ---------------------------------------------------------------------------

@runtime_checkable
class JevDecider(Protocol):
    def __call__(
        self,
        frames_rgb: list[np.ndarray],
        fps: float,
        t_ms: int,
    ) -> dict[str, float]: ...
    # t_ms is context for logging/tests only; a real decider MUST NOT use it to decide.


# ---------------------------------------------------------------------------
# validate_probs (§7.3)
# ---------------------------------------------------------------------------

def validate_probs(raw: object) -> dict[str, float]:
    """
    Validates that raw is a dict with exactly the HAZARDS key set and all values
    are int|float (not bool), finite, and in [0.0, 1.0].
    Raises ValueError("invalid_output: <reason>") on any violation.
    Returns {h: float(raw[h]) for h in HAZARDS} (HAZARDS order).
    """
    if not isinstance(raw, dict):
        raise ValueError(f"invalid_output: expected dict, got {type(raw).__name__}")
    expected = set(HAZARDS)
    actual = set(raw.keys())
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        parts = []
        if missing:
            parts.append(f"missing keys: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected keys: {sorted(extra)}")
        raise ValueError(f"invalid_output: {'; '.join(parts)}")
    result: dict[str, float] = {}
    for h in HAZARDS:
        v = raw[h]
        if isinstance(v, bool):
            raise ValueError(f"invalid_output: {h} is bool, must be int or float")
        if not isinstance(v, (int, float)):
            raise ValueError(f"invalid_output: {h} has type {type(v).__name__}, must be numeric")
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError(f"invalid_output: {h}={v} is not finite")
        if not (0.0 <= fv <= 1.0):
            raise ValueError(f"invalid_output: {h}={fv} out of range [0.0, 1.0]")
        result[h] = fv
    return result


# ---------------------------------------------------------------------------
# evaluate_tick (§5.2)
# ---------------------------------------------------------------------------

def evaluate_tick(
    decider: JevDecider,
    cap,  # cv2.VideoCapture
    tick_id: int,
    t_ms: int,
    cfg: GateConfig,
) -> TickRecord:
    """
    Evaluates one JEV tick.  Pure timing starts at entry.
    """
    t0 = time.perf_counter()
    window_start_ms = max(0, t_ms - cfg.window_ms)

    # Step 2: sample timestamps, read BGR frames
    timestamps = sample_timestamps(window_start_ms, t_ms, cfg.jev_sample_fps)
    frames_bgr, _ = read_frames_at(cap, timestamps, rgb=False)

    # Step 3: no frames -> error, decider NOT called
    if not frames_bgr:
        return TickRecord(
            tick_id=tick_id,
            t_ms=t_ms,
            window_start_ms=window_start_ms,
            num_frames=0,
            status="error",
            error="no_frames",
            probs=None,
            thumbs=[],
            jev_time_s=time.perf_counter() - t0,
        )

    # Step 4: resize + convert BGR -> RGB (NO JPEG encoding for decider, D-17)
    frames_resized: list[np.ndarray] = [resize_max_side(f, cfg.jev_max_side) for f in frames_bgr]
    frames_rgb: list[np.ndarray] = [f[:, :, ::-1].copy() for f in frames_resized]

    # Step 5: thumbnails at {0, n//2, n-1} of resized BGR frames
    n = len(frames_resized)
    thumb_indices = sorted({0, n // 2, n - 1})
    thumbs = [make_thumb(frames_resized[i]) for i in thumb_indices]

    # Step 6: call decider — never catches BaseException / KeyboardInterrupt / SystemExit
    try:
        raw = decider(frames_rgb, cfg.jev_sample_fps, t_ms)
    except Exception as e:
        return TickRecord(
            tick_id=tick_id,
            t_ms=t_ms,
            window_start_ms=window_start_ms,
            num_frames=n,
            status="error",
            error=f"{type(e).__name__}: {e}",
            probs=None,
            thumbs=[],
            jev_time_s=time.perf_counter() - t0,
        )

    # Step 7: validate
    try:
        probs = validate_probs(raw)
    except ValueError as e:
        return TickRecord(
            tick_id=tick_id,
            t_ms=t_ms,
            window_start_ms=window_start_ms,
            num_frames=n,
            status="error",
            error=str(e),
            probs=None,
            thumbs=[],
            jev_time_s=time.perf_counter() - t0,
        )

    # Step 8: success
    return TickRecord(
        tick_id=tick_id,
        t_ms=t_ms,
        window_start_ms=window_start_ms,
        num_frames=n,
        status="ok",
        probs=probs,
        error=None,
        thumbs=thumbs,
        jev_time_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# OneJevDecider (Appendix A, in-process, lazy-imports torch/transformers)
# ---------------------------------------------------------------------------

class OneJevDecider:
    label = "onejev"

    def __init__(self, model_path: str) -> None:
        # Lazy import — torch and transformers are NOT imported at module level (I-5)
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError(
                "OneJev requires CUDA (--jev onejev); no CUDA device is available"
            )
        if not os.path.isdir(model_path):
            raise RuntimeError(f"OneJev model not found: {model_path}")

        self._processor = AutoProcessor.from_pretrained(model_path)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to("cuda")
        model.eval()
        self._model = model
        self._torch = torch

        # Slot ids: "A" and "B" must each be exactly one token
        tok = self._processor.tokenizer
        id_a_list = tok.encode("A", add_special_tokens=False)
        id_b_list = tok.encode("B", add_special_tokens=False)
        if len(id_a_list) != 1 or len(id_b_list) != 1:
            raise RuntimeError("slot letters are not single tokens")
        self._id_a = id_a_list[0]
        self._id_b = id_b_list[0]

    def __call__(
        self,
        frames_rgb: list[np.ndarray],
        fps: float,
        t_ms: int,
    ) -> dict[str, float]:
        """
        One forward pass per hazard question (Appendix A §4-6).
        Returns {hazard: p_yes} for all five hazards.
        """
        import torch
        from PIL import Image

        # Media prep (Appendix A §2)
        frames_pil = [Image.fromarray(f) for f in frames_rgb]
        n = len(frames_pil)
        try:
            from transformers.image_utils import VideoMetadata  # type: ignore
            video_metadata = [VideoMetadata(
                total_num_frames=n,
                fps=fps,
                frames_indices=list(range(n)),
                duration=n / fps if fps > 0 else 1.0,
            )]
        except ImportError:
            video_metadata = None

        # State block (Appendix A §3)
        import json as _json
        state = {
            "task": (
                "Warehouse CCTV safety monitoring. The clip is a short (about 3 seconds) view from a "
                "fixed indoor camera showing workers, forklifts and stored goods. Judge only what is "
                "visible in the clip. Answer yes only if the described condition is actually occurring "
                "or clearly about to occur, not merely because the relevant objects are present."
            ),
            "camera": "cam-1",  # per camera, e.g. "cam-2: loading dock, wide view of two bays"
            "clip": "<video:1>",
        }
        state_str = f"<state>\n{_json.dumps(state, indent=2)}\n</state>"

        results: dict[str, float] = {}
        for h in HAZARD_QUESTIONS:
            question_text = HAZARD_QUESTIONS[h]

            # Build prompt (Appendix A §4)
            user_text = (
                f"{state_str}\n\nQuestion: Is the statement true, or is the answer to the question yes?\n\n"
                f"Options:\nA. yes: {question_text}\nB. no: the statement is false / the answer is no\n\n"
                "Answer with one letter: A, B."
            )
            system_prompt = (
                "Apply the question to the state. "
                "Choose exactly one of the listed options. "
                "Respond with only its uppercase letter, with no explanation or reasoning."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": user_text}]},
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            proc_kwargs: dict = {
                "text": [text],
                "videos": [frames_pil],
                "videos_kwargs": {"do_sample_frames": False, "cap_pixels_per_frame": True},
                "return_tensors": "pt",
            }
            if video_metadata is not None:
                proc_kwargs["video_metadata"] = video_metadata
            inputs = self._processor(**proc_kwargs)
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

            # Readout (Appendix A §5)
            with torch.no_grad():
                out = self._model(**inputs, logits_to_keep=1)
            slot_logits = out.logits[0, -1][[self._id_a, self._id_b]].float()
            # softmax with T=1.0
            slot_logits = slot_logits - slot_logits.max()
            exp = torch.exp(slot_logits)
            p_yes = float((exp[0] / exp.sum()).clamp(0.0, 1.0))
            results[h] = p_yes

        return results