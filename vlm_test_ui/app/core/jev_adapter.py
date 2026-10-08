from __future__ import annotations
"""
jev_adapter.py - JEV decider interface, validate_probs, evaluate_tick, OneJevDecider.

Imports: copy, dataclasses, typing, time, math, logging, cv2, numpy, app.core.*.
No torch/transformers at module level (lazy, I-5).
"""
import copy
import math
import logging
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
# Shared-prefix helpers (design_v3 §5.2) — pure, testable without a model
# ---------------------------------------------------------------------------
MARKER: str = "@@JEV_PREFIX_MARK@@"


def split_prompt_suffixes(prefix_render: str, full_renders: list[str]) -> list[str]:
    """
    prefix_render is apply_chat_template output for messages whose user text ends
    with MARKER, rendered with add_generation_prompt=False. full_renders are the
    five per-hazard renders (add_generation_prompt=True). Every full render MUST
    start with prefix_render[:index_of_MARKER]; otherwise raise
    ValueError('prompt prefix mismatch'). Returns the five string suffixes
    full_renders[i][len(prefix_cut):].
    """
    idx = prefix_render.find(MARKER)
    if idx < 0:
        raise ValueError("prompt prefix mismatch")
    prefix_cut = prefix_render[:idx]
    suffixes: list[str] = []
    for full in full_renders:
        if not full.startswith(prefix_cut):
            raise ValueError("prompt prefix mismatch")
        suffixes.append(full[len(prefix_cut):])
    return suffixes


def suffix_position_ids(last_prefix_positions: "torch.Tensor", suffix_len: int) -> "torch.Tensor":
    """
    last_prefix_positions: [3] positions of the final prefix token (captured per
    design_v3 §5.2 step 3). Returns [3, 1, suffix_len] where element [a, 0, j] =
    last_prefix_positions[a] + 1 + j.
    """
    import torch

    start = last_prefix_positions.reshape(3, 1, 1)
    steps = torch.arange(
        1, suffix_len + 1,
        device=last_prefix_positions.device,
        dtype=last_prefix_positions.dtype,
    ).view(1, 1, -1)
    return start + steps


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

        # fp32 LM-head upcast (design_v3 §5.1; port of qev engine._upcast_head):
        # bf16 logits at magnitude 25-50 quantize to 0.125 steps (~3% probability
        # steps, exact ties), so the head is untied (clone) and computed in fp32
        # while the rest of the forward stays bf16.
        head = model.get_output_embeddings()
        if head is None or not hasattr(head, "weight"):
            self.head_dtype = "model"
            logging.getLogger(__name__).warning(
                "model has no separable output head; logits stay in the model dtype"
            )
        else:
            self.head_dtype = "float32"
            head.weight = torch.nn.Parameter(head.weight.detach().clone().to(torch.float32), requires_grad=False)
            if getattr(head, "bias", None) is not None:
                head.bias = torch.nn.Parameter(head.bias.detach().clone().to(torch.float32), requires_grad=False)
            head.register_forward_pre_hook(lambda module, args: (args[0].to(torch.float32),) + tuple(args[1:]))
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
        self.mode = "shared"  # §5.2 path; "fullpass" = the v2 five-pass path (DEV-B2 only)

    def __call__(
        self,
        frames_rgb: list[np.ndarray],
        fps: float,
        t_ms: int,
        *,
        order: str = "ab",
    ) -> dict[str, float]:
        """
        Shared-prefix tick evaluation (design_v3 §5.2): one processor call and one
        video prefill per tick, five text-only suffix branches off deep-copied caches.
        mode == "fullpass" keeps the v2 five-pass path (DEV-B2 equivalence only).
        Returns {hazard: p_yes} for all five hazards.
        """
        if order not in ("ab", "ba"):
            raise ValueError(f"invalid order: {order!r}")
        if self.mode == "fullpass":
            return self._call_fullpass(frames_rgb, fps, t_ms, order=order)
        return self._call_shared(frames_rgb, fps, t_ms, order=order)

    # ------------------------------------------------------------------
    # Prompt construction (verbatim v2 texts; "ba" swaps the two option lines)
    # ------------------------------------------------------------------

    def _build_messages(self, order: str) -> tuple[str, str, list[list[dict]]]:
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
        system_prompt = (
            "Apply the question to the state. "
            "Choose exactly one of the listed options. "
            "Respond with only its uppercase letter, with no explanation or reasoning."
        )

        messages_list: list[list[dict]] = []
        for h in HAZARD_QUESTIONS:
            question_text = HAZARD_QUESTIONS[h]
            if order == "ba":
                options_block = (
                    f"Options:\nA. no: the statement is false / the answer is no\nB. yes: {question_text}\n\n"
                )
            else:
                options_block = (
                    f"Options:\nA. yes: {question_text}\nB. no: the statement is false / the answer is no\n\n"
                )
            user_text = (
                f"{state_str}\n\nQuestion: Is the statement true, or is the answer to the question yes?\n\n"
                f"{options_block}"
                "Answer with one letter: A, B."
            )
            messages_list.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": user_text}]},
            ])
        return state_str, system_prompt, messages_list

    def _prepare_media(self, frames_rgb: list[np.ndarray], fps: float):
        from PIL import Image

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
        return frames_pil, video_metadata

    def _decoder_layers(self) -> list:
        model = self._model
        for path in ("model.language_model.layers", "language_model.layers", "model.model.layers"):
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return list(obj)
        raise RuntimeError("could not locate language-model decoder layers for position capture")

    # ------------------------------------------------------------------
    # §5.2 shared-prefix path
    # ------------------------------------------------------------------

    def _call_shared(
        self,
        frames_rgb: list[np.ndarray],
        fps: float,
        t_ms: int,
        *,
        order: str = "ab",
    ) -> dict[str, float]:
        import torch

        frames_pil, video_metadata = self._prepare_media(frames_rgb, fps)
        state_str, system_prompt, messages_list = self._build_messages(order)

        # Step 1: renders + suffix split (video placeholder stays inside the prefix)
        prefix_user_text = f"{state_str}{MARKER}"
        prefix_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prefix_user_text}]},
        ]
        prefix_render = self._processor.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=False
        )
        full_renders = [
            self._processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        suffixes = split_prompt_suffixes(prefix_render, full_renders)
        prefix_cut = prefix_render[: prefix_render.index(MARKER)]

        # Step 2: one processor call + one prefill on the cut prefix render
        # (no explicit position_ids — the model computes them internally)
        proc_kwargs: dict = {
            "text": [prefix_cut],
            "videos": [frames_pil],
            "videos_kwargs": {"do_sample_frames": False, "cap_pixels_per_frame": True},
            "return_tensors": "pt",
        }
        if video_metadata is not None:
            proc_kwargs["video_metadata"] = video_metadata
        inputs = self._processor(**proc_kwargs)
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        device = inputs["input_ids"].device
        P_exp = inputs["input_ids"].shape[1]

        # Step 3: position capture — authentic OneJev mm_engine implementation (mm_engine.py:537-548).
        # Qwen-VL computes 3D multimodal RoPE positions via get_rope_index.
        computed_positions = None
        rope_func = getattr(getattr(self._model, "model", None), "get_rope_index", None) or getattr(self._model, "get_rope_index", None)
        if rope_func is not None:
            try:
                import inspect
                sig_params = inspect.signature(rope_func).parameters
                rope_kwargs = {}
                for k in ("input_ids", "mm_token_type_ids", "attention_mask", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
                    if k in inputs and inputs[k] is not None:
                        if k in sig_params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values()):
                            rope_kwargs[k] = inputs[k]
                rope_res = rope_func(**rope_kwargs)
                if isinstance(rope_res, tuple):
                    computed_positions = rope_res[0]
                elif hasattr(rope_res, "shape"):
                    computed_positions = rope_res
            except Exception as exc:
                logging.getLogger(__name__).warning("get_rope_index failed (%s); falling back to hook/linear", exc)

        # Forward pre-hooks record layer position_ids (also supports test stubs)
        recorded: list = []
        handles = []
        try:
            decoder_layers = self._decoder_layers()
        except Exception:
            decoder_layers = []
        for layer in decoder_layers:
            def _recorder(module, args, kwargs, _store=recorded):
                pos = kwargs.get("position_ids")
                if pos is None and len(args) > 2 and isinstance(args[2], torch.Tensor):
                    pos = args[2]
                elif pos is None and len(args) > 1 and isinstance(args[1], torch.Tensor):
                    if args[1].dim() == 3 or args[1].shape[-1] == inputs["input_ids"].shape[1]:
                        pos = args[1]
                if pos is not None:
                    _store.append(pos)
                return None
            handles.append(layer.register_forward_pre_hook(_recorder, with_kwargs=True))
        try:
            with torch.no_grad():
                out = self._model(**inputs, use_cache=True, return_dict=True, logits_to_keep=1)
            prefix_cache = out.past_key_values
        finally:
            for handle in handles:
                handle.remove()

        if recorded and recorded[-1] is not None:
            rec = recorded[-1]
            if rec.dim() == 3:
                last_prefix_positions = rec[:, 0, -1]
            elif rec.dim() == 2:
                last_prefix_positions = rec[:, -1]
            else:
                last_prefix_positions = rec.view(3, -1)[:, -1]
        elif computed_positions is not None:
            if computed_positions.dim() == 3:
                last_prefix_positions = computed_positions[:, 0, -1]
            elif computed_positions.dim() == 2:
                last_prefix_positions = computed_positions[:, -1]
            else:
                last_prefix_positions = computed_positions.view(3, -1)[:, -1]
        else:
            last_prefix_positions = torch.tensor([P_exp - 1, P_exp - 1, P_exp - 1], device=device, dtype=torch.long)
        del out

        # Step 4: five text-only suffix branches off deep-copied caches
        slot_ids = [self._id_a, self._id_b] if order == "ab" else [self._id_b, self._id_a]
        tok = self._processor.tokenizer
        results: dict[str, float] = {}
        for hazard, suffix_text in zip(HAZARD_QUESTIONS, suffixes):
            suffix_ids = tok.encode(suffix_text, add_special_tokens=False)
            cache = copy.deepcopy(prefix_cache)
            try:
                with torch.no_grad():
                    out = self._model(
                        input_ids=torch.tensor([suffix_ids], device=device),
                        attention_mask=torch.ones(
                            (1, P_exp + len(suffix_ids)), dtype=torch.long, device=device
                        ),
                        position_ids=suffix_position_ids(last_prefix_positions, len(suffix_ids)),
                        past_key_values=cache,
                        use_cache=True,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                slot_logits = out.logits[0, -1][slot_ids].float()
            finally:
                del cache
            # Step 5: slot readout and softmax (T=1.0; fp32 head from §5.1)
            slot_logits = slot_logits - slot_logits.max()
            exp = torch.exp(slot_logits)
            p_yes = float((exp[0] / exp.sum()).clamp(0.0, 1.0))
            results[hazard] = p_yes
        del prefix_cache
        return results

    # ------------------------------------------------------------------
    # v2 five-pass path (kept for the DEV-B2 equivalence measurement only)
    # ------------------------------------------------------------------

    def _call_fullpass(
        self,
        frames_rgb: list[np.ndarray],
        fps: float,
        t_ms: int,
        *,
        order: str = "ab",
    ) -> dict[str, float]:
        import torch

        frames_pil, video_metadata = self._prepare_media(frames_rgb, fps)
        _state_str, _system_prompt, messages_list = self._build_messages(order)

        results: dict[str, float] = {}
        for h, messages in zip(HAZARD_QUESTIONS, messages_list):
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
            slot_ids = [self._id_a, self._id_b] if order == "ab" else [self._id_b, self._id_a]
            slot_logits = out.logits[0, -1][slot_ids].float()
            # softmax with T=1.0
            slot_logits = slot_logits - slot_logits.max()
            exp = torch.exp(slot_logits)
            p_yes = float((exp[0] / exp.sum()).clamp(0.0, 1.0))
            results[h] = p_yes

        return results