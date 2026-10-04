import os
import sys

# Ensure Hugging Face cache is strictly on F: drive
os.environ["HF_HOME"] = r"F:\URCA_PROJECTS\.hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"F:\URCA_PROJECTS\.hf_cache\hub"
os.environ["TRANSFORMERS_CACHE"] = r"F:\URCA_PROJECTS\.hf_cache\transformers"

# Add current workspace to sys.path so app and vlm_test_ui can be imported if needed
workspace_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
vlm_test_ui_path = os.path.join(workspace_root, "vlm_test_ui")
if vlm_test_ui_path not in sys.path:
    sys.path.insert(0, vlm_test_ui_path)

import torch
from transformers import AutoProcessor, BitsAndBytesConfig

MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"

def main():
    print("=" * 60)
    print("PHASE 0.1 SPIKE: INSPECT QWEN3-VL")
    print("=" * 60)

    # 1. Resolve Class
    QwenVLClass = None
    resolved_class_name = None
    try:
        from transformers import Qwen3VLForConditionalGeneration as QwenVLClass
        resolved_class_name = "Qwen3VLForConditionalGeneration"
    except ImportError:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLClass
            resolved_class_name = "Qwen2_5_VLForConditionalGeneration"
        except ImportError:
            from transformers import AutoModelForVision2Seq as QwenVLClass
            resolved_class_name = "AutoModelForVision2Seq"

    print(f"[*] Resolved class target: {resolved_class_name}")

    # 2. Load Processor
    print(f"[*] Loading AutoProcessor with local_files_only=True...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    print(f"[+] Processor loaded: {type(processor)}")

    # 3. Load Model (4-bit BitsAndBytes)
    print(f"[*] Loading model with local_files_only=True and 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )

    model = QwenVLClass.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    print(f"[+] Model loaded successfully!")
    print(f"type(model): {type(model)}")
    print(f"type(model.config): {type(model.config)}")
    print(f"Primary model device: {next(model.parameters()).device}")

    # 4. Attribute checks
    print("\n" + "=" * 60)
    print("ATTRIBUTE CHECKS")
    print("=" * 60)

    # model.model.language_model.layers or equivalent
    has_lang_layers = False
    try:
        layers = model.model.language_model.layers
        print(f"hasattr(model, 'model.language_model.layers'): True, count={len(layers)}")
        has_lang_layers = True
    except Exception as e:
        print(f"hasattr(model, 'model.language_model.layers'): False ({e})")
        # search named_modules for 'layers'
        matching = []
        for name, mod in model.named_modules():
            if name.endswith("layers") or ".layers" in name:
                matching.append(name)
        print(f"Matching module paths for 'layers' in named_modules: {matching[:10]}")
        # Check standard model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            print(f"Found model.model.layers: count={len(model.model.layers)}")

    # model.config attributes
    print(f"hasattr(model.config, 'video_token_id'): {hasattr(model.config, 'video_token_id')}")
    if hasattr(model.config, 'video_token_id'):
        print(f"  video_token_id value: {model.config.video_token_id}")
    else:
        video_attrs = [a for a in dir(model.config) if "video" in a.lower() or "token_id" in a.lower()]
        print(f"  Relevant config attributes: {video_attrs}")

    print(f"hasattr(model.config, 'vision_start_token_id'): {hasattr(model.config, 'vision_start_token_id')}")
    if hasattr(model.config, 'vision_start_token_id'):
        print(f"  vision_start_token_id: {model.config.vision_start_token_id}")
    print(f"hasattr(model.config, 'vision_end_token_id'): {hasattr(model.config, 'vision_end_token_id')}")
    if hasattr(model.config, 'vision_end_token_id'):
        print(f"  vision_end_token_id: {model.config.vision_end_token_id}")

    # get_video_features
    print(f"hasattr(model, 'get_video_features'): {hasattr(model, 'get_video_features')} (callable: {callable(getattr(model, 'get_video_features', None))})")
    if hasattr(model, 'get_video_features'):
        import inspect
        print(f"get_video_features signature: {inspect.signature(model.get_video_features)}")

    # get_rope_index
    print(f"hasattr(model, 'get_rope_index'): {hasattr(model, 'get_rope_index')} (callable: {callable(getattr(model, 'get_rope_index', None))})")
    if hasattr(model, "model"):
        print(f"hasattr(model.model, 'get_rope_index'): {hasattr(model.model, 'get_rope_index')} (callable: {callable(getattr(model.model, 'get_rope_index', None))})")
        rope_methods = [m for m in dir(model.model) if "rope" in m.lower() or "position" in m.lower()]
        print(f"model.model rope/position methods: {rope_methods}")

    # 5. Forward Pass Spike with dummy 2-frame input
    print("\n" + "=" * 60)
    print("FORWARD PASS & KV CACHE INSPECTION")
    print("=" * 60)

    from qwen_vl_utils import process_vision_info
    import cv2
    from PIL import Image
    video_path = os.path.join(vlm_test_ui_path, "examples", "synthetic_test.mp4")
    print(f"Using video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames_pil = []
    while cap.isOpened() and len(frames_pil) < 2:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(frame_rgb))
    cap.release()
    print(f"Read {len(frames_pil)} frames via cv2 into PIL Images.")

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames_pil},
            {"type": "text", "text": "Describe the video."}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print(f"Input keys: {list(inputs.keys())}")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")

    # Test get_video_features
    try:
        vid_feat = model.get_video_features(
            inputs["pixel_values_videos"],
            video_grid_thw=inputs["video_grid_thw"]
        )
        print(f"vid_feat type: {type(vid_feat)}")
        if hasattr(vid_feat, "keys"):
            print(f"vid_feat keys: {vid_feat.keys()}")
            for k in vid_feat.keys():
                val = vid_feat[k]
                if hasattr(val, "shape"):
                    print(f"  vid_feat[{k}] shape: {val.shape}, dtype={val.dtype}")
                elif isinstance(val, (list, tuple)):
                    print(f"  vid_feat[{k}] len: {len(val)}")
                    for idx, item in enumerate(val):
                        if hasattr(item, "shape"):
                            print(f"    item {idx} shape: {item.shape}, dtype={item.dtype}")
                        else:
                            print(f"    item {idx} type: {type(item)}")
        elif isinstance(vid_feat, (list, tuple)):
            print(f"get_video_features returned list/tuple of len {len(vid_feat)}")
            print(f"  item 0 shape: {vid_feat[0].shape}, dtype: {vid_feat[0].dtype}")
        elif hasattr(vid_feat, "shape"):
            print(f"get_video_features return shape: {vid_feat.shape}, dtype: {vid_feat.dtype}")
    except Exception as e:
        print(f"Error calling get_video_features: {e}")

    # Inspect how model.model.forward handles pixel_values_videos
    import inspect
    model_forward_src = inspect.getsource(model.model.forward)
    print("\n[+] model.model forward visual feature handling lines:")
    for line in model_forward_src.split("\n"):
        if any(term in line for term in ["get_video_features", "pixel_values_videos", "deepstack", "visual"]):
            print(f"  {line.strip()}")

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)

    print(f"\ntype(outputs): {type(outputs)}")
    print(f"outputs keys / len: {dir(outputs) if hasattr(outputs, '__dict__') else len(outputs)}")

    past_key_values = getattr(outputs, "past_key_values", None)
    if past_key_values is None and isinstance(outputs, (tuple, list)) and len(outputs) > 1:
        past_key_values = outputs[1]

    print(f"type(past_key_values): {type(past_key_values)}")

    # Check cache layout
    if hasattr(past_key_values, "layers"):
        print(f"past_key_values has .layers attribute (count={len(past_key_values.layers)})")
        layer_0 = past_key_values.layers[0]
        print(f"type(layer_0): {type(layer_0)}")
        print(f"hasattr(layer_0, 'keys'): {hasattr(layer_0, 'keys')}")
        print(f"hasattr(layer_0, 'values'): {hasattr(layer_0, 'values')}")
        if hasattr(layer_0, 'keys'):
            print(f"layer_0.keys shape: {layer_0.keys.shape}, dtype={layer_0.keys.dtype}, device={layer_0.keys.device}")
            print(f"layer_0.values shape: {layer_0.values.shape}, dtype={layer_0.values.dtype}, device={layer_0.values.device}")
    elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        print("past_key_values has .key_cache and .value_cache (DynamicCache standard)")
        print(f"num layers in key_cache: {len(past_key_values.key_cache)}")
        print(f"key_cache[0] shape: {past_key_values.key_cache[0].shape}")
        print(f"value_cache[0] shape: {past_key_values.value_cache[0].shape}")
        # check if .layers also exists or not
        print(f"dir(past_key_values): {[a for a in dir(past_key_values) if not a.startswith('_')]}")
    elif isinstance(past_key_values, (tuple, list)):
        print(f"past_key_values is a tuple/list of length {len(past_key_values)}")
        print(f"layer 0 type: {type(past_key_values[0])}")
        if isinstance(past_key_values[0], (tuple, list)):
            print(f"past_key_values[0][0] (key) shape: {past_key_values[0][0].shape}")
            print(f"past_key_values[0][1] (val) shape: {past_key_values[0][1].shape}")
    else:
        print(f"dir(past_key_values): {[a for a in dir(past_key_values) if not a.startswith('_')]}")

    # Check 3D RoPE index
    print("\n" + "=" * 60)
    print("3D-RoPE / POSITION ID INSPECTION")
    print("=" * 60)
    rope_func = None
    if hasattr(model, "get_rope_index"):
        rope_func = model.get_rope_index
    elif hasattr(model, "model") and hasattr(model.model, "get_rope_index"):
        rope_func = model.model.get_rope_index

    if rope_func:
        import inspect
        sig = inspect.signature(rope_func)
        print(f"get_rope_index signature: {sig}")
        try:
            # Check calling with input_ids, video_grid_thw, mm_token_type_ids
            rope_kwargs = {
                "input_ids": inputs["input_ids"],
                "video_grid_thw": inputs.get("video_grid_thw", None),
            }
            if "mm_token_type_ids" in inputs:
                rope_kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"]
            if "image_grid_thw" in inputs:
                rope_kwargs["image_grid_thw"] = inputs["image_grid_thw"]

            rope_res = rope_func(**rope_kwargs)
            if isinstance(rope_res, tuple):
                print(f"get_rope_index returned tuple of len {len(rope_res)}")
                for i, r in enumerate(rope_res):
                    if r is not None and hasattr(r, 'shape'):
                        print(f"  rope_res[{i}] shape: {r.shape}, dtype={r.dtype}")
                    else:
                        print(f"  rope_res[{i}]: {type(r)}")
            elif hasattr(rope_res, 'shape'):
                print(f"get_rope_index return shape: {rope_res.shape}, dtype={rope_res.dtype}")
            else:
                print(f"get_rope_index return: {type(rope_res)}")
        except Exception as e:
            print(f"Error calling get_rope_index with kwargs: {e}")
    else:
        print("No get_rope_index found on model or model.model.")

    # Device placement inspection
    print("\n" + "=" * 60)
    print("DEVICE PLACEMENT INSPECTION")
    print("=" * 60)
    devices_seen = set()
    for name, param in model.named_parameters():
        devices_seen.add(str(param.device))
    print(f"Parameters distributed across devices: {devices_seen}")

    print("\n[+] Spike execution finished successfully.")

if __name__ == "__main__":
    main()
