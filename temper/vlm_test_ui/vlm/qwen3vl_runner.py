from __future__ import annotations
import os
import sys

# Ensure Hugging Face cache is strictly on F: drive
os.environ["HF_HOME"] = r"F:\URCA_PROJECTS\.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"F:\URCA_PROJECTS\.hf_cache\hub"
os.environ["TRANSFORMERS_CACHE"] = r"F:\URCA_PROJECTS\.hf_cache\transformers"

import copy
import json
import re
import tempfile
import cv2
import numpy as np
import torch
from PIL import Image

from app.core.types import VideoChunk, VLMResult, InvalidVLMReply
from app.config import DEFAULT_CONFIG
from .infinipot_v.kvcache_utils import process_kv_cache
from .vision_common import preprocess_video_frames, video_position_ids_3d

MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"


class Qwen3VLRunner:
    """
    Callable matching VLMCallable: runner(chunk) -> VLMResult.
    Maintains a continuous InfiniPot-V compressed KV cache across sequential video chunks,
    bypassing disk I/O and running stateful autoregressive visual inference.
    """
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        prompt: str = DEFAULT_CONFIG.chunk_prompt,
        num_frames: int = 16,
        max_new_tokens: int = 160,
        load_in_4bit: bool = True,
        memory_budget_frames: int = 16,
        compress_to_frames: int = 8,
        compression_method: str = "infinipot-v",
        use_infinipot: bool = True,
    ):
        self.prompt = prompt
        self.num_frames = num_frames
        self.max_new_tokens = max_new_tokens
        self.model_path = model_path
        self.memory_budget_frames = memory_budget_frames
        self.compress_to_frames = compress_to_frames
        self.compression_method = compression_method
        self.use_infinipot = use_infinipot

        # Session state (cleared by reset())
        self.past_key_values = None       # persistent DynamicCache
        self.position_ids_tracker = None  # running position offset
        self.system_embeds_cached = None  # system/prompt tokens embedded once
        self.frames_seen = 0
        self.system_size = 0
        self.token_per_frame = None
        self._pos_offset = 0              # next free M-RoPE position (all 3 dims)
        self.last_video_grid_thw = None   # telemetry: grid of last ingested chunk

        from transformers import AutoProcessor
        print(f"[*] Loading AutoProcessor for {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

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
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
        else:
            self.model = QwenVLClass.from_pretrained(
                model_path,
                dtype="auto",
                device_map="auto",
                local_files_only=True,
            )

        # Device introspection & logging
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            print(f"[*] Model device map: {device_map}")
            devices = set(device_map.values())
            if "cpu" in devices and len(devices) > 1:
                print("[!] WARNING: CPU offloading active — InfiniPot-V cache operations may incur PCIe transfer overhead.")
        else:
            param_devices = {p.device.type for p in self.model.parameters()}
            print(f"[*] Model parameter devices: {param_devices}")
            if "cpu" in param_devices and "cuda" in param_devices:
                print("[!] WARNING: CPU offloading active — InfiniPot-V cache operations may incur PCIe transfer overhead.")

        print("[+] Model loaded successfully. Performing warm-up...")
        self._warmup()

    def reset(self) -> None:
        """
        Call at the start of a new video session. Clears any persistent KV cache state.
        Idempotent: calling twice in succession is a safe no-op.
        """
        self.past_key_values = None
        self.position_ids_tracker = None
        self.system_embeds_cached = None
        self.frames_seen = 0
        self.system_size = 0
        self.token_per_frame = None
        self._pos_offset = 0
        self.last_video_grid_thw = None

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

    def _append_chunk_to_cache(self, chunk: VideoChunk) -> None:
        """
        Encodes chunk frame pixels directly into the vision tower and appends to the
        persistent KV cache via one forward pass. Triggers compression immediately if budget is exceeded.
        """
        device = next(self.model.parameters()).device

        # Shared preprocessing (identical to the HERMES path): no hidden frame
        # re-sampling, same resize policy, true fps metadata.
        proc_inputs, grid = preprocess_video_frames(self.processor, chunk.frames, chunk.fps)
        self.last_video_grid_thw = grid
        pixel_values_videos = proc_inputs["pixel_values_videos"].to(device)
        video_grid_thw = proc_inputs["video_grid_thw"].to(device)

        with torch.inference_mode():
            feat = self.model.get_video_features(pixel_values_videos, video_grid_thw=video_grid_thw)
            video_embeds = feat.pooler_output[0]  # [tokens_in_this_chunk, hidden_dim]

        # 3D M-RoPE positions for the video tokens. Without explicit position_ids,
        # an inputs_embeds forward falls back to flat 1D positions and the model
        # loses temporal/spatial video structure.
        vision_config = getattr(self.model.config, "vision_config", None)
        sys_pos = None
        if self.past_key_values is None:
            # First chunk: embed system prompt once and prepend to first video block
            sys_text = "<|im_start|>system\nYou are a helpful industrial surveillance and safety monitoring visual assistant.<|im_end|>\n"
            sys_ids = self.processor.tokenizer(sys_text, return_tensors="pt").input_ids.to(device)
            self.system_size = sys_ids.shape[1]
            self.system_embeds_cached = self.model.model.get_input_embeddings()(sys_ids)
            self.token_per_frame = max(1, video_embeds.shape[0] // max(1, len(chunk.frames)))
            block_inputs_embeds = torch.cat([self.system_embeds_cached, video_embeds.unsqueeze(0)], dim=1)
            sys_pos = torch.arange(self.system_size, dtype=torch.float32).unsqueeze(0).expand(3, -1)
            base = float(self.system_size)
        else:
            # Subsequent chunks: only visual embeddings (system tokens already in cache)
            block_inputs_embeds = video_embeds.unsqueeze(0)
            base = float(self._pos_offset)

        video_pos = video_position_ids_3d(grid, base, vision_config=vision_config)
        if sys_pos is not None:
            position_ids = torch.cat([sys_pos, video_pos], dim=1).unsqueeze(1)  # [3, 1, L]
        else:
            position_ids = video_pos.unsqueeze(1)                               # [3, 1, L]
        position_ids = position_ids.to(device)

        # Run forward pass appending to existing persistent cache
        with torch.inference_mode():
            outputs = self.model(
                inputs_embeds=block_inputs_embeds,
                past_key_values=self.past_key_values,
                use_cache=True,
                position_ids=position_ids,
            )

        self.past_key_values = outputs.past_key_values
        self._pos_offset = int(position_ids.max().item()) + 1
        self.frames_seen += len(chunk.frames)

        # Trigger compression immediately after cache growth, before next chunk
        self._maybe_compress(self.memory_budget_frames, self.compress_to_frames)

    def _maybe_compress(self, memory_budget_frames: int, compress_to_frames: int) -> None:
        """
        Checks if persistent visual cache exceeds memory_budget_frames.
        If exceeded, applies InfiniPot-V compression down to compress_to_frames.
        """
        if self.past_key_values is None or self.token_per_frame is None or self.token_per_frame <= 0:
            return

        current_seq_len = self.past_key_values.layers[0].keys.shape[2]
        vision_length = current_seq_len - self.system_size
        current_frame_num = vision_length // self.token_per_frame

        if current_frame_num > memory_budget_frames:
            # Clamp vision_length to the nearest complete-frame boundary.
            # Qwen3-VL uses dynamic resolution so individual chunks may produce
            # slightly varying token counts, leaving a fractional tail in the cache.
            # We floor to the nearest multiple of token_per_frame, discarding at most
            # one partial frame, so process_kv_cache's divisibility pre-condition holds.
            aligned_vision_len = current_frame_num * self.token_per_frame
            aligned_seq_len = self.system_size + aligned_vision_len
            if aligned_seq_len < current_seq_len:
                # Trim trailing fractional tokens from every layer in-place
                for layer in self.past_key_values.layers:
                    layer.keys = layer.keys[:, :, :aligned_seq_len, :]
                    layer.values = layer.values[:, :, :aligned_seq_len, :]

            self.past_key_values, _ = process_kv_cache(
                past_key_values=self.past_key_values,
                model=self.model,
                system_size=self.system_size,
                inst_size=0,
                token_per_frame=self.token_per_frame,
                compress_frame_num=compress_to_frames,
                method=self.compression_method,
            )

    def _generate_from_cache(self, prompt_text: str) -> str:
        """
        Autoregressively decodes a response from the current KV cache.
        Non-mutation guarantee (approach a): We deep-copy self.past_key_values into a throwaway
        gen_cache. Autoregressive decoding tokens are appended only to gen_cache and discarded,
        guaranteeing self.past_key_values remains unmutated for subsequent chunks.
        """
        device = next(self.model.parameters()).device
        gen_cache = copy.deepcopy(self.past_key_values)

        gen_messages = [{"role": "user", "content": prompt_text}]
        prompt_formatted = self.processor.apply_chat_template(gen_messages, tokenize=False, add_generation_prompt=True)
        prompt_inputs = self.processor.tokenizer(prompt_formatted, return_tensors="pt").to(device)
        curr_input_ids = prompt_inputs.input_ids

        # Continue M-RoPE positions after the last ingested video token, so the
        # question tokens are RoPE-aligned with the (3D-positioned) cache.
        prompt_len = curr_input_ids.shape[1]
        offset = self._pos_offset
        pos_1d = torch.arange(offset, offset + prompt_len, device=device, dtype=torch.float32)
        position_ids = pos_1d.view(1, 1, -1).expand(3, 1, -1).contiguous()

        generated_tokens = []
        eos_token_id = getattr(self.processor.tokenizer, 'eos_token_id', 151645)

        for _ in range(self.max_new_tokens):
            with torch.inference_mode():
                gen_out = self.model(
                    input_ids=curr_input_ids,
                    past_key_values=gen_cache,
                    use_cache=True,
                    position_ids=position_ids,
                )
            logits = gen_out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            gen_cache = gen_out.past_key_values
            position_ids = position_ids[:, :, -1:] + 1
            token_id = next_token.item()
            if token_id in (eos_token_id, 151645, 151643):
                break
            generated_tokens.append(token_id)
            curr_input_ids = next_token

        self._pos_offset = offset + prompt_len + len(generated_tokens)

        return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def _ask(self, content_item: dict, question_text: str) -> str:
        """Legacy ask method used for warmup and fallback."""
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
                do_sample=False,
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def _legacy_call(self, chunk: VideoChunk) -> VLMResult:
        """Legacy stateless per-chunk generation via temporary video files."""
        with tempfile.TemporaryDirectory(dir=DEFAULT_CONFIG.temp_dir) as tmp:
            if len(chunk.frames) == 1:
                path = os.path.join(tmp, "frame.jpg")
                cv2.imwrite(path, cv2.cvtColor(chunk.frames[0], cv2.COLOR_RGB2BGR))
                item = {"type": "image", "image": path}
            else:
                path = os.path.join(tmp, "chunk.mp4")
                self._write_clip(chunk.frames, chunk.fps, path)
                n = min(self.num_frames, len(chunk.frames))
                n = max(n - (n % 2), 2)
                item = {"type": "video", "video": path, "nframes": n}

            raw = self._ask(item, self.prompt)
        return self._parse(raw)

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
        if not self.use_infinipot:
            return self._legacy_call(chunk)
        self._append_chunk_to_cache(chunk)
        raw = self._generate_from_cache(self.prompt)
        return self._parse(raw)

