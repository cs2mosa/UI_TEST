from __future__ import annotations
import os
from dataclasses import dataclass, field

@dataclass
class AppConfig:
    # Project paths
    project_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    hf_cache_dir: str = r"F:\URCA_PROJECTS\.hf_cache"
    temp_dir: str = r"F:\URCA_PROJECTS\.tmp"
    cache_dir: str = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), ".cache")

    # Chunk settings
    default_chunk_s: float = 5.0
    default_overlap_s: float = 1.0
    default_vlm_fps: float = 4.0
    include_partial_last_window: bool = True

    # Ground truth matching thresholds
    tol: float = 0.25
    high_thr: float = 0.66

    # Model settings
    qwen_model_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    hermes_model_path: str = (
        r"F:\URCA_PROJECTS\.hf_cache\hub"
        r"\models--Qwen--Qwen3-VL-4B-Instruct"
        r"\snapshots\ebb281ec70b05090aa6165b016eac8ec08e71b17"
    )
    max_new_tokens: int = 160
    num_frames: int = 16

    # HERMES streaming settings
    hermes_kv_size: int = 2000          # max visual KV tokens/layer; 2000 for 6 GB VRAM
    hermes_sample_fps: float = 0.5      # temporal position IDs frame rate
    hermes_encode_chunk_size: int = 8   # frames encoded in one GPU call (VRAM safety)
    hermes_load_in_4bit: bool = True    # NF4 4-bit quant required for 6 GB VRAM

    # Default VLM prompt as defined in Design Doc §15.3
    chunk_prompt: str = (
        "You are a safety monitor reviewing a short clip from a fixed warehouse or industrial "
        "CCTV camera. Workers, forklifts, pallets and racking may be present.\n\n"
        "Look carefully for these hazards:\n"
        "1. Pedestrian-vehicle: a person on foot in the path of, or very close to, a moving "
        "forklift, pallet jack or truck.\n"
        "2. Missing PPE: a worker in an active work area without a hi-vis vest or hard hat.\n"
        "3. Blocked access: a fire exit, emergency equipment or marked aisle obstructed by "
        "goods or equipment.\n"
        "4. Unsafe load: stacked or carried goods leaning, overhanging, raised too high while "
        "moving, or falling.\n"
        "5. Fall or spill: a person on the floor, or a spill, debris or object on the walking "
        "or driving surface.\n"
        "Also note any other clearly unsafe behaviour, such as running, climbing racking or "
        "riding on forks.\n\n"
        "Rules:\n"
        "- Describe only what is clearly visible. Do not guess or invent details. A hazard "
        "needs visible evidence, not just the presence of a person or a forklift.\n"
        "- If there is no hazard, say briefly what is happening and give a low score.\n"
        "- If there are several hazards, describe the most severe first and score that one.\n"
        "- Caption: one or two short sentences, under 40 words, saying who or what is involved, "
        "where they are relative to the danger, and why it is unsafe.\n\n"
        "Score the most severe situation in the clip from 0.0 to 1.0:\n"
        "0.0-0.2 routine activity, no hazard; 0.3-0.5 unsafe behaviour without immediate "
        "danger; 0.6-0.8 clear safety violation or near-miss; "
        "0.9-1.0 accident or serious danger in progress.\n\n"
        "Reply with JSON only, with no markdown fences and no other text, using exactly this form: "
        '{"caption": "<text>", "score": <number from 0 to 1>}\n'
        "Do not use double quotes or braces inside the caption.\n\n"
        "Format examples (illustration of style only, do not copy their content):\n"
        '{"caption": "A worker in a hi-vis vest and hard hat walks along a marked aisle while a '
        'forklift is parked nearby.", "score": 0.1}\n'
        '{"caption": "A worker without a hard hat stands in the forklift lane as a loaded forklift '
        'approaches within two metres.", "score": 0.8}\n'
        '{"caption": "A stack of boxes collapses into the aisle beside a worker who is knocked off '
        'balance.", "score": 0.9}'
    )

    # Optional label map for categorical ground truths
    label_map: dict[str, float] = field(default_factory=lambda: {
        "safe": 0.0,
        "normal": 0.0,
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8,
        "hazard": 0.9,
        "critical": 1.0,
    })

    # UI Theme Palette (Modern Dark Mode)
    color_bg: str = "#0f172a"          # slate-900
    color_surface: str = "#1e293b"     # slate-800
    color_surface_alt: str = "#334155" # slate-700
    color_border: str = "#475569"      # slate-600
    color_text: str = "#f8fafc"        # slate-50
    color_text_muted: str = "#94a3b8"  # slate-400
    color_primary: str = "#3b82f6"     # blue-500
    color_primary_hover: str = "#2563eb"
    
    # Severity & Status colors
    color_safe: str = "#22c55e"        # green-500 (low score)
    color_warning: str = "#eab308"     # yellow-500 (medium score)
    color_hazard: str = "#ef4444"      # red-500 (high score)
    
    # GT Match colors
    color_match: str = "#10b981"       # emerald-500
    color_miss: str = "#ef4444"        # red-500
    color_fp: str = "#f97316"          # orange-500
    color_off: str = "#facc15"         # amber-400

DEFAULT_CONFIG = AppConfig()
