from __future__ import annotations
import os
import sys

# Ensure Hugging Face cache is strictly on F: drive
os.environ["HF_HOME"] = r"F:\URCA_PROJECTS\.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"F:\URCA_PROJECTS\.hf_cache\hub"
os.environ["TRANSFORMERS_CACHE"] = r"F:\URCA_PROJECTS\.hf_cache\transformers"

import json
import re
import tempfile
import cv2
import numpy as np
import torch

from app.core.types import VideoChunk, VLMResult, InvalidVLMReply
from app.config import DEFAULT_CONFIG

MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"


class Qwen3VLRunner:
    """
    Callable matching VLMCallable: runner(chunk) -> VLMResult.
    Runs a quantized vision-language model with temporary clip generation and structured JSON prompting.
    """
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        prompt: str = DEFAULT_CONFIG.chunk_prompt,
        num_frames: int = 16,
        max_new_tokens: int = 160,
        load_in_4bit: bool = True,
    ):
        self.prompt = prompt
        self.num_frames = num_frames
        self.max_new_tokens = max_new_tokens
        self.model_path = model_path

        from transformers import AutoProcessor
        print(f"[*] Loading AutoProcessor for {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)

        # Attempt to import specialized Qwen3 VL class, falling back to Qwen2.5 or Auto
        try:
            from transformers import Qwen3VLForConditionalGeneration as QwenVLClass
        except ImportError:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLClass
            except ImportError:
                from transformers import AutoModelForVision2Seq as QwenVLClass

        print(f"[*] Loading model {model_path} on GPU (4-bit BitsAndBytes)...")
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["visual", "lm_head"],
            )
            self.model = QwenVLClass.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                dtype=torch.bfloat16,
            )
        else:
            self.model = QwenVLClass.from_pretrained(
                model_path,
                dtype="auto",
                device_map="auto",
            )

        print("[+] Model loaded successfully. Performing warm-up...")
        self._warmup()

    def _warmup(self):
        """Warm-up call on one dummy frame so first real chunk latency is not inflated."""
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            with tempfile.TemporaryDirectory() as tmp:
                p = os.path.join(tmp, "warmup.jpg")
                cv2.imwrite(p, dummy)
                item = {"type": "image", "image": p}
                self._ask(item, "Is this an image?")
            print("[+] Warmup complete.")
        except Exception as e:
            print(f"[!] Warmup skipped ({e})")

    def _ask(self, content_item: dict, question_text: str) -> str:
        from qwen_vl_utils import process_vision_info
        messages = [{
            "role": "user",
            "content": [content_item, {"type": "text", "text": question_text}],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Design Doc §15.4: deterministic reproducibility
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    @staticmethod
    def _write_clip(frames: list[np.ndarray], fps: float, path: str):
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()

    @staticmethod
    def _parse(raw: str) -> VLMResult:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise InvalidVLMReply(raw)
        try:
            d = json.loads(m.group(0))
            caption = str(d.get("caption", "")).strip()
            score = float(d.get("score", 0.0))
            return VLMResult(caption=caption, score=score)
        except Exception:
            raise InvalidVLMReply(raw)

    def __call__(self, chunk: VideoChunk) -> VLMResult:
        with tempfile.TemporaryDirectory(dir=DEFAULT_CONFIG.temp_dir) as tmp:
            if len(chunk.frames) == 1:
                path = os.path.join(tmp, "frame.jpg")
                cv2.imwrite(path, cv2.cvtColor(chunk.frames[0], cv2.COLOR_RGB2BGR))
                item = {"type": "image", "image": path}
            else:
                path = os.path.join(tmp, "chunk.mp4")
                self._write_clip(chunk.frames, chunk.fps, path)
                n = min(self.num_frames, len(chunk.frames))
                n = max(n - (n % 2), 2)  # Even number of frames >= 2
                item = {"type": "video", "video": path, "nframes": n}

            raw = self._ask(item, self.prompt)
        return self._parse(raw)
