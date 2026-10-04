import os
import sys
import torch
import cv2
from PIL import Image

os.environ["HF_HOME"] = r"F:\URCA_PROJECTS\.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"F:\URCA_PROJECTS\.hf_cache\hub"
os.environ["TRANSFORMERS_CACHE"] = r"F:\URCA_PROJECTS\.hf_cache\transformers"

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", local_files_only=True)
dummy_frames = [Image.new("RGB", (224, 224), (i * 50, 100, 150)) for i in range(2)]

# Test image_processor directly
out = processor.image_processor(videos=[dummy_frames], return_tensors="pt")
print("image_processor output keys:", list(out.keys()))
for k, v in out.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
