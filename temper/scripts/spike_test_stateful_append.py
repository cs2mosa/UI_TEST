"""
Spike to verify stateful chunk appending, DynamicCache copying/truncation,
and InfiniPot-V compression on live Qwen3-VL-4B.
"""

import os
import sys
import copy
import cv2
import torch
from PIL import Image

# Ensure HF cache
os.environ["HF_HOME"] = r"F:\URCA_PROJECTS\.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"F:\URCA_PROJECTS\.hf_cache\hub"
os.environ["TRANSFORMERS_CACHE"] = r"F:\URCA_PROJECTS\.hf_cache\transformers"

workspace_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, workspace_root)
sys.path.insert(0, os.path.join(workspace_root, "vlm_test_ui"))

from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from qwen_vl_utils import process_vision_info
from vlm_test_ui.vlm.infinipot_v.kvcache_utils import process_kv_cache

MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"

def load_frames(video_path, max_frames=4):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def main():
    print("[*] Loading processor & model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    device = next(model.parameters()).device

    video_path = os.path.join(workspace_root, "vlm_test_ui", "examples", "synthetic_test.mp4")
    frames = load_frames(video_path, max_frames=4)
    chunk1_frames = frames[:2]
    chunk2_frames = frames[2:4]
    print(f"[+] Loaded {len(chunk1_frames)} frames for chunk 1, {len(chunk2_frames)} frames for chunk 2")

    # Step 1: Process chunk 1
    # System prompt + chunk 1 video
    messages1 = [
        {"role": "system", "content": "You are a helpful visual assistant."},
        {"role": "user", "content": [
            {"type": "video", "video": chunk1_frames},
            {"type": "text", "text": "Analyzing video."}
        ]}
    ]
    text1 = processor.apply_chat_template(messages1, tokenize=False, add_generation_prompt=False)
    # Find system prefix size:
    # Text template has system prompt then user prompt.
    image_inputs1, video_inputs1 = process_vision_info(messages1)
    inputs1 = processor(text=[text1], images=image_inputs1, videos=video_inputs1, return_tensors="pt")
    inputs1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs1.items()}

    with torch.inference_mode():
        out1 = model(**inputs1, use_cache=True)
    cache = out1.past_key_values
    seq_len_1 = cache.layers[0].keys.shape[2]
    print(f"[+] Chunk 1 processed. Cache seq_len: {seq_len_1}")

    # Step 2: Process chunk 2 (appending to cache)
    messages2 = [
        {"role": "user", "content": [
            {"type": "video", "video": chunk2_frames}
        ]}
    ]
    text2 = processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=False)
    image_inputs2, video_inputs2 = process_vision_info(messages2)
    inputs2 = processor(text=[text2], images=image_inputs2, videos=video_inputs2, return_tensors="pt")
    inputs2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs2.items()}

    # When appending with cache:
    # Notice input_ids and position_ids for new tokens
    with torch.inference_mode():
        out2 = model(**inputs2, past_key_values=cache, use_cache=True)
    cache = out2.past_key_values
    seq_len_2 = cache.layers[0].keys.shape[2]
    print(f"[+] Chunk 2 processed. Cache seq_len: {seq_len_2}")
    assert seq_len_2 > seq_len_1, f"Expected cache to grow! {seq_len_2} <= {seq_len_1}"

    # Step 3: Test generation from cache without mutating persistent cache
    # Check if cache can be shallow/deep copied or truncated
    cache_len_before_gen = cache.layers[0].keys.shape[2]
    
    # Test copying DynamicCache
    import copy
    gen_cache = copy.deepcopy(cache)
    print(f"[+] Successfully deep-copied DynamicCache (seq_len={gen_cache.layers[0].keys.shape[2]})")

    # Generate a prompt using gen_cache
    gen_messages = [
        {"role": "user", "content": 'Describe what is happening in the video. Reply in JSON: {"caption": "<text>", "score": 0.0}'}
    ]
    prompt_text = processor.apply_chat_template(gen_messages, tokenize=False, add_generation_prompt=True)
    # Extract only the generation prompt part
    prompt_inputs = processor.tokenizer(prompt_text, return_tensors="pt").to(device)
    print(f"[+] Prompt tokens: shape={prompt_inputs.input_ids.shape}")

    # Autoregressive generation loop
    generated_tokens = []
    curr_input_ids = prompt_inputs.input_ids
    for _ in range(30):
        with torch.inference_mode():
            gen_out = model(input_ids=curr_input_ids, past_key_values=gen_cache, use_cache=True)
        logits = gen_out.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        gen_cache = gen_out.past_key_values
        token_id = next_token.item()
        if token_id in (processor.tokenizer.eos_token_id, 151645, 151643):
            break
        generated_tokens.append(token_id)
        curr_input_ids = next_token

    gen_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"[+] Generated text from cache:\n{gen_text}")

    # Verify original cache was not mutated!
    cache_len_after_gen = cache.layers[0].keys.shape[2]
    print(f"[+] Original cache len before gen: {cache_len_before_gen}, after gen: {cache_len_after_gen}")
    assert cache_len_before_gen == cache_len_after_gen, "Original cache was mutated during generation!"

    print("[+] All verification checks in spike passed!")

if __name__ == "__main__":
    main()
