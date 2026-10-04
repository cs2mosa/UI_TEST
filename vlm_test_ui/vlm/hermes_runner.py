"""
Stateful streaming VLM runner backed by the HERMES Qwen3-VL engine.

Implements the VLMCallable protocol:  runner(chunk: VideoChunk) -> VLMResult

Session lifecycle:
  - call start_session() before the first chunk of a new video
  - call __call__(chunk) for every chunk in temporal order
  - call end_session() after the last chunk  (BatchWorker does this)
  - BatchWorker.run() auto-detects the presence of start/end_session
"""
from __future__ import annotations

import json
import re
import numpy as np
import torch

from app.core.types import VideoChunk, VLMResult, InvalidVLMReply


class HermesStreamingRunner:
    """
    Wraps Qwen3VLHermes engine as a VLMCallable.

    Args:
        model_path: HuggingFace model path or local cache path.
        prompt: Safety assessment prompt sent to the model after each chunk.
        kv_size: Max visual KV tokens retained per layer (2000 for 6 GB VRAM).
        sample_fps: Frame sampling rate for temporal position IDs.
        load_in_4bit: 4-bit NF4 quantization (required for 6 GB VRAM).
        max_new_tokens: Max tokens the model generates per chunk.
        encode_chunk_size: Frames encoded in one GPU call (limits peak VRAM).
    """

    def __init__(
        self,
        model_path: str = (
            r"F:\URCA_PROJECTS\.hf_cache\hub"
            r"\models--Qwen--Qwen3-VL-4B-Instruct"
            r"\snapshots\ebb281ec70b05090aa6165b016eac8ec08e71b17"
        ),
        prompt: str = (
            'Analyse this scene for industrial safety hazards. '
            'Reply with ONLY valid JSON: {"caption": "<concise description>", '
            '"score": <0.0-1.0>}. '
            'Score 0=no hazard, 1=imminent danger. '
            'No extra text, no markdown fences.'
        ),
        kv_size: int = 2000,
        sample_fps: float = 0.5,
        load_in_4bit: bool = True,
        max_new_tokens: int = 160,
        encode_chunk_size: int = 8,
    ):
        self.prompt = prompt
        self.kv_size = kv_size
        self.sample_fps = sample_fps
        self.max_new_tokens = max_new_tokens
        self.encode_chunk_size = encode_chunk_size
        self._session_active = False

        from vlm.hermes_engine import load_hermes_model
        self.model, self.processor = load_hermes_model(
            model_path=model_path,
            kv_size=kv_size,
            streaming=True,
            sample_fps=sample_fps,
            load_in_4bit=load_in_4bit,
        )
        self._warmup()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _warmup(self):
        """Encode a 2-frame stub to JIT-compile CUDA kernels before real inference."""
        try:
            self.start_session()
            dummy = np.zeros((2, 64, 64, 3), dtype=np.uint8)
            video_tensor = torch.from_numpy(dummy)
            self.model.encode_video_chunk(video_tensor)
            self.model.predict_and_compress()
            self.end_session()
            print("[+] HERMES warmup complete.")
        except Exception as exc:
            print(f"[!] HERMES warmup skipped: {exc}")
            self.end_session()

    def start_session(self):
        """Initialize HERMES KV cache for a new video. Call before first chunk."""
        self.model.clear_cache()
        self.model.encode_init_prompt()
        self._session_active = True

    def end_session(self):
        """Release KV cache. Safe to call even if no session is active."""
        self.model.clear_cache()
        self._session_active = False

    # ------------------------------------------------------------------
    # VLMCallable implementation
    # ------------------------------------------------------------------

    def __call__(self, chunk: VideoChunk) -> VLMResult:
        """
        Process one VideoChunk through the HERMES streaming pipeline.

        If called without a prior start_session() (e.g. image mode), a
        session is automatically started.
        """
        if not self._session_active:
            self.start_session()

        # [N, H, W, 3] uint8 RGB  →  torch tensor
        frames_array = np.stack(chunk.frames, axis=0)
        video_tensor = torch.from_numpy(frames_array)  # stays on CPU until encode_video_chunk

        # Encode in fixed-size sub-chunks to control peak VRAM
        total = video_tensor.shape[0]
        for start in range(0, total, self.encode_chunk_size):
            sub = video_tensor[start : start + self.encode_chunk_size]
            self.model.encode_video_chunk(sub, frame_fps=chunk.fps)

        # Safety-aware KV compression
        self.model.predict_and_compress()

        # Generate structured JSON response
        input_text = {
            "question": self.prompt,
            "prompt": self.model.get_prompt(self.prompt),
        }
        raw = self.model.question_answering(
            input_text, max_new_tokens=self.max_new_tokens
        )

        return self._parse(raw)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(raw: str) -> VLMResult:
        """
        Extract JSON {"caption": ..., "score": ...} from model output.
        Raises InvalidVLMReply if the output cannot be parsed.
        """
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not m:
            raise InvalidVLMReply(raw)
        try:
            d = json.loads(m.group(0))
            caption = str(d.get("caption", "")).strip()
            score = float(d.get("score", 0.0))
            return VLMResult(caption=caption, score=score)
        except Exception:
            raise InvalidVLMReply(raw)
