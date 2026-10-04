import os
import sys
import torch
import cv2
from PIL import Image

os.environ["HF_HOME"] = r"F:\URCA_PROJECTS\.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"F:\URCA_PROJECTS\.hf_cache\hub"
os.environ["TRANSFORMERS_CACHE"] = r"F:\URCA_PROJECTS\.hf_cache\transformers"

from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

def main():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", local_files_only=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    device = next(model.parameters()).device

    # Create 2 frames
    frames1 = [Image.new("RGB", (224, 224), (255, 0, 0)), Image.new("RGB", (224, 224), (0, 255, 0))]
    frames2 = [Image.new("RGB", (224, 224), (0, 0, 255)), Image.new("RGB", (224, 224), (255, 255, 0))]

    # Get video features for chunk 1
    msg1 = [{"role": "user", "content": [{"type": "video", "video": frames1}]}]
    _, v1 = process_vision_info(msg1)
    inp1 = processor(text=["<|video_pad|>"], videos=v1, return_tensors="pt")
    pix1 = inp1["pixel_values_videos"].to(device)
    thw1 = inp1["video_grid_thw"].to(device)

    with torch.inference_mode():
        feat1 = model.get_video_features(pix1, video_grid_thw=thw1)
        video_embeds1 = feat1.pooler_output[0] # [tokens, 2560]
    
    print(f"video_embeds1 shape: {video_embeds1.shape}")
    token_per_frame = video_embeds1.shape[0] // len(frames1)
    print(f"token_per_frame: {token_per_frame}")

    # Embed system tokens
    sys_text = "<|im_start|>system\nYou are a helpful safety monitor.<|im_end|>\n"
    sys_ids = processor.tokenizer(sys_text, return_tensors="pt").input_ids.to(device)
    sys_embeds = model.model.get_input_embeddings()(sys_ids)
    system_size = sys_ids.shape[1]
    print(f"system_size: {system_size}")

    # First block: system + vision
    block1_embeds = torch.cat([sys_embeds, video_embeds1.unsqueeze(0)], dim=1)
    print(f"block1_embeds shape: {block1_embeds.shape}")

    with torch.inference_mode():
        out1 = model(inputs_embeds=block1_embeds, use_cache=True)
    cache = out1.past_key_values
    seq1 = cache.layers[0].keys.shape[2]
    print(f"seq1: {seq1} (expected: {system_size + video_embeds1.shape[0]})")
    assert seq1 == system_size + video_embeds1.shape[0]

    # Chunk 2: ONLY vision embeds
    msg2 = [{"role": "user", "content": [{"type": "video", "video": frames2}]}]
    _, v2 = process_vision_info(msg2)
    inp2 = processor(text=["<|video_pad|>"], videos=v2, return_tensors="pt")
    pix2 = inp2["pixel_values_videos"].to(device)
    thw2 = inp2["video_grid_thw"].to(device)

    with torch.inference_mode():
        feat2 = model.get_video_features(pix2, video_grid_thw=thw2)
        video_embeds2 = feat2.pooler_output[0]

    print(f"video_embeds2 shape: {video_embeds2.shape}")
    block2_embeds = video_embeds2.unsqueeze(0)

    with torch.inference_mode():
        out2 = model(inputs_embeds=block2_embeds, past_key_values=cache, use_cache=True)
    cache = out2.past_key_values
    seq2 = cache.layers[0].keys.shape[2]
    print(f"seq2: {seq2} (expected: {seq1 + video_embeds2.shape[0]})")
    assert seq2 == seq1 + video_embeds2.shape[0]

    vis_len = seq2 - system_size
    print(f"vis_len: {vis_len}, divisible by token_per_frame ({token_per_frame})? {vis_len % token_per_frame == 0}")
    assert vis_len % token_per_frame == 0

    print("[+] SUCCESS! inputs_embeds direct vision feeding works cleanly and maintains exact frame alignment!")

if __name__ == "__main__":
    main()
