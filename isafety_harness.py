from __future__ import annotations
"""
isafety_harness.py - iSafetyBench scoring harness for JEV threshold calibration.

One clip = one calibration item (design_v3 §5.3): frames are sampled with the
runtime tick numerics (sample_timestamps / read_frames_at / resize_max_side from
app.core.frame_sampler), scored once by the OneJevDecider, and one labeled row is
emitted per hazard (Appendix C). --dump-logits additionally writes the raw slot
logits per (item, hazard) — the PROBE-1 fp32 readout probe (DEV-A1).

DEV-only (D-32): this script loads a model on GPU. The agent never runs it without
explicit per-run developer authorisation; results count only when the developer
clears them.

Prompt-hash (D-28, glossary): SHA-256 over json.dumps(HAZARD_QUESTIONS, sort_keys=True)
+ the state-block text + the system prompt. The state-block and system-prompt texts
are pinned copies of the OneJevDecider prompt construction; any drift on either side
changes the hash and voids calibration.

Staging (design_v3 §7): Phase A = scoring + --dump-logits probe. Phase B adds
--decider-mode {shared,fullpass}. Phase C adds --null-input and --order both.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "vlm_test_ui")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from app.core.jev_adapter import HAZARD_QUESTIONS, OneJevDecider
from app.core.frame_sampler import read_frames_at, resize_max_side, sample_timestamps

# ---------------------------------------------------------------------------
# Pinned prompt texts (verbatim copies of OneJevDecider's construction, for
# the prompt hash only — the harness never renders prompts itself)
# ---------------------------------------------------------------------------
_STATE = {
    "task": (
        "Warehouse CCTV safety monitoring. The clip is a short (about 3 seconds) view from a "
        "fixed indoor camera showing workers, forklifts and stored goods. Judge only what is "
        "visible in the clip. Answer yes only if the described condition is actually occurring "
        "or clearly about to occur, not merely because the relevant objects are present."
    ),
    "camera": "cam-1",  # per camera, e.g. "cam-2: loading dock, wide view of two bays"
    "clip": "<video:1>",
}
_STATE_BLOCK_TEXT = f"<state>\n{json.dumps(_STATE, indent=2)}\n</state>"
_SYSTEM_PROMPT = (
    "Apply the question to the state. "
    "Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

_DEFAULT_JEV_MODEL = (
    r"F:\URCA_PROJECTS\.hf_cache\hub"
    r"\models--OmniJev--OneJev-0.8B"
    r"\snapshots\c3939d8bf4cad34549a2b13bbb6aee9bcb6afee8"
)


def compute_prompt_hash() -> str:
    payload = json.dumps(HAZARD_QUESTIONS, sort_keys=True) + _STATE_BLOCK_TEXT + _SYSTEM_PROMPT
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_model_path(model_path: str, repo_id: str = "OmniJev/OneJev-0.8B") -> str:
    """Ensures model snapshot exists locally; finds existing snapshot or downloads it automatically."""
    if os.path.isdir(model_path) and any(os.scandir(model_path)):
        return model_path

    # Check if any snapshot directory exists under the model's repo cache
    for parent in [
        os.path.dirname(model_path),
        os.path.join(BASE_DIR, ".hf_cache", "hub", "models--OmniJev--OneJev-0.8B", "snapshots"),
        os.path.join(os.environ.get("HF_HOME", ""), "hub", "models--OmniJev--OneJev-0.8B", "snapshots"),
    ]:
        if parent and os.path.isdir(parent):
            matches = [os.path.join(parent, d) for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
            for m in matches:
                if any(os.scandir(m)):
                    print(f"[*] Found existing OneJev snapshot at: {m}", flush=True)
                    return m

    print(f"[*] Local OneJev model not found at: {model_path}", flush=True)
    print(f"[*] Downloading '{repo_id}' via huggingface_hub...", flush=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[!] huggingface_hub not installed, cannot download model automatically.", flush=True)
        return model_path

    hf_cache = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not hf_cache:
        hf_cache = os.path.join(BASE_DIR, ".hf_cache")

    os.makedirs(hf_cache, exist_ok=True)
    downloaded_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=hf_cache,
    )
    print(f"[+] OneJev model ready at: {downloaded_dir}", flush=True)
    return downloaded_dir


# ---------------------------------------------------------------------------
# Slot-logit capture (PROBE-1)
# ---------------------------------------------------------------------------

class _SlotLogitCapture:
    """
    Captures (logit_a, logit_b) straight from the LM head output during a normal
    decider call. The fp32 head (design_v3 §5.1) is the last compute op before the
    runtime readout, so a forward hook on it sees the exact logits the slot softmax
    consumes — no prompt/forward duplication. OneJevDecider performs one head pass
    per hazard question in HAZARD_QUESTIONS order, so capture order maps 1:1.
    """

    def __init__(self, decider: OneJevDecider) -> None:
        self._decider = decider
        self._handle = None
        self.logits: list[tuple[float, float]] = []

    def __enter__(self) -> "_SlotLogitCapture":
        head = self._decider._model.get_output_embeddings()
        capture = self

        def _hook(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            row = tensor[0, -1]
            capture.logits.append((
                float(row[capture._decider._id_a].item()),
                float(row[capture._decider._id_b].item()),
            ))

        self._handle = head.register_forward_hook(_hook)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Mapping / items
# ---------------------------------------------------------------------------

def load_mapping(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError(f"mapping must be a JSON object, got {type(mapping).__name__}")
    required = {"version", "hazards", "unmapped_hazard_categories", "routine_categories", "notes"}
    missing = required - set(mapping)
    if missing:
        raise ValueError(f"mapping missing keys: {sorted(missing)}")
    if set(mapping["hazards"]) != set(HAZARD_QUESTIONS):
        raise ValueError("mapping hazards must be exactly the five HAZARD_QUESTIONS keys")
    return mapping


def load_items(dataset_dir: str, mapping: dict) -> list[dict]:
    """
    Delegates to isafety_loader.load_items (C0 deliverable, design_v3 §7; pinned
    interface: CalibrationItem = {item_id, video_path, hazard_labels, tags, camera}).
    Imported lazily so this file imports cleanly before C0 lands.
    """
    from isafety_loader import load_items as _load_items
    return _load_items(dataset_dir, mapping)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def make_null_items() -> list[dict]:
    """
    §5.3 --null-input: uniform mid-gray frame set, 12 frames @ 320x320, measuring the
    context-free yes-prior (Zhao et al. 2021). One item — the dataset rows carry no
    camera and the loader surfaces no resolution buckets, so the '(or one, if none)'
    branch applies. hazard_labels 'null' on all five hazards → five rows emitted.
    """
    frames = [np.full((320, 320, 3), 128, dtype=np.uint8) for _ in range(12)]
    return [{
        "item_id": "null_0",
        "video_path": None,
        "frames": frames,
        "hazard_labels": {h: "null" for h in HAZARD_QUESTIONS},
        "tags": [],
        "camera": None,
        "source": "null_input",
    }]


def score_item(decider, item: dict, sample_fps: float, max_side: int, order_probe: bool = False) -> dict:
    """
    One clip = one item (§5.3): sample at the given fps over the full clip, resize
    to max_side, one decider call (plus a second order='ba' call when order_probe).
    Returns rows/logits or a skip reason. Duration comes from the stream metadata
    (annotation formats are unknown until C0 inspects them; nothing here reads
    annotation fields beyond CalibrationItem). Null items carry precomputed frames.
    """
    if item.get("frames") is not None:
        frames_rgb = item["frames"]
        duration_ms = 0
    else:
        cap = cv2.VideoCapture(item["video_path"])
        if not cap.isOpened():
            return {"skipped": "video_open_failed"}
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_ms = int(round(frame_count / src_fps * 1000.0)) if src_fps > 0 and frame_count > 0 else 0
        if duration_ms <= 0:
            cap.release()
            return {"skipped": "zero_duration"}

        timestamps = sample_timestamps(0, duration_ms, sample_fps)
        frames_rgb, _ = read_frames_at(cap, timestamps, rgb=True)
        cap.release()
        if not frames_rgb:
            return {"skipped": "no_frames"}
        frames_rgb = [resize_max_side(f, max_side) for f in frames_rgb]
    n = len(frames_rgb)

    with _SlotLogitCapture(decider) as capture:
        t0 = time.perf_counter()
        probs = decider(frames_rgb, sample_fps, 0)
        score_s = time.perf_counter() - t0
        probs_ba = decider(frames_rgb, sample_fps, 0, order="ba") if order_probe else None
    expected_captures = len(HAZARD_QUESTIONS) * (2 if order_probe else 1)
    # In shared-prefix mode, the prefix forward fires the LM head once per decider call before the hazard branches
    n_h = len(HAZARD_QUESTIONS)
    if len(capture.logits) == (n_h + 1) * (2 if order_probe else 1):
        if order_probe:
            capture.logits = capture.logits[1 : n_h + 1] + capture.logits[n_h + 2 : 2 * n_h + 2]
        else:
            capture.logits = capture.logits[1 : n_h + 1]

    if len(capture.logits) != expected_captures:
        return {"skipped": f"logit_capture_mismatch ({len(capture.logits)} of {expected_captures})"}

    source = item.get("source", "isafety")
    labels: dict = item.get("hazard_labels") or {}
    rows: list[dict] = []
    logits: list[dict] = []

    def _emit(p_yes: float, hazard: str, order: str, cap_index: int) -> None:
        label = labels.get(hazard)
        if label is None:
            return  # unmappable hazard-on-item pairs are omitted, never assumed negative
        rows.append({
            "source": "order_probe" if order == "ba" else source,
            "item_id": item["item_id"],
            "hazard": hazard,
            "p_yes": float(p_yes),
            "label": label,
            "tags": item.get("tags", []),
            "order": order,
            "camera": item.get("camera"),
            "meta": {"duration_ms": duration_ms, "n_frames": n},
        })
        logit_a, logit_b = capture.logits[cap_index]
        logits.append({
            "item_id": item["item_id"],
            "hazard": hazard,
            "order": order,
            "logit_a": logit_a,
            "logit_b": logit_b,
            "p_yes": float(p_yes),
        })

    for offset, (hazard, p_yes) in enumerate(probs.items()):
        _emit(p_yes, hazard, "ab", offset)
    if probs_ba is not None:
        for offset, (hazard, p_yes) in enumerate(probs_ba.items()):
            _emit(p_yes, hazard, "ba", len(HAZARD_QUESTIONS) + offset)
    return {"rows": rows, "logits": logits, "score_s": score_s, "n_frames": n}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Score iSafetyBench clips with OneJev for JEV threshold calibration"
    )
    parser.add_argument("--dataset-dir", required=True, help="Local iSafetyBench snapshot directory")
    parser.add_argument("--mapping", required=True, help="isafety_mapping.json worksheet")
    parser.add_argument("--model-path", default=os.environ.get("JEV_MODEL_PATH", _DEFAULT_JEV_MODEL),
                        help="OneJev-0.8B snapshot (default: JEV_MODEL_PATH env or the pinned snapshot)")
    parser.add_argument("--out", default="labeled_scores.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N items (after the subset rule)")
    parser.add_argument("--dump-logits", default=None, help="Also write raw slot logits per (item, hazard) to this JSONL")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Tick-analogue sampling fps")
    parser.add_argument("--max-side", type=int, default=640, help="Tick-analogue max frame side")
    parser.add_argument("--seed", type=int, default=42, help="Recorded in run_meta (consumed by Phase C)")
    parser.add_argument("--decider-mode", choices=["shared", "fullpass"], default="shared",
                        help="shared = the design_v3 §5.2 shared-prefix path; "
                             "fullpass = the v2 five-pass path (DEV-B2 equivalence only)")
    parser.add_argument("--null-input", action="store_true",
                        help="add gray-frame null items (context-free yes-prior rows, §5.3)")
    parser.add_argument("--order", choices=["ab", "both"], default="ab",
                        help="'both' adds a second order='ba' decider call per item (order-probe rows, §5.3)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        mapping = load_mapping(args.mapping)
    except (OSError, ValueError) as exc:
        print(f"[!] mapping error: {exc}", file=sys.stderr)
        return 2

    items = load_items(args.dataset_dir, mapping)
    if not items:
        print("FAILED: no items survived the subset rule", file=sys.stderr)
        return 3
    if args.limit is not None:
        items = items[: args.limit]
    if args.null_input:
        items = items + make_null_items()
    print(f"[*] {len(items)} calibration items to score (dataset: {args.dataset_dir})"
          + (" + null-input" if args.null_input else "")
          + (", order-probe on" if args.order == "both" else ""))

    args.model_path = ensure_model_path(args.model_path, repo_id="OmniJev/OneJev-0.8B")
    decider = OneJevDecider(args.model_path)
    decider.mode = args.decider_mode
    print(f"[*] OneJev loaded; head_dtype={decider.head_dtype}; decider_mode={decider.mode}")

    run_meta = {
        "type": "run_meta",
        "prompt_hash": compute_prompt_hash(),
        "model_path": args.model_path,
        "head_dtype": decider.head_dtype,
        "gate_config": {"jev_sample_fps": args.sample_fps, "jev_max_side": args.max_side},
        "dataset_dir": args.dataset_dir,
        "mapping_version": mapping["version"],
        "seed": args.seed,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    logits_file = open(args.dump_logits, "w", encoding="utf-8") if args.dump_logits else None
    if logits_file is not None:
        logits_file.write(json.dumps({"type": "logits_meta", **run_meta}, separators=(",", ":")) + "\n")

    n_scored, n_skipped = 0, 0
    skip_reasons: dict[str, int] = {}
    try:
        with open(args.out, "w", encoding="utf-8") as out:
            out.write(json.dumps(run_meta, separators=(",", ":")) + "\n")
            for i, item in enumerate(items):
                result = score_item(decider, item, args.sample_fps, args.max_side,
                                    order_probe=(args.order == "both"))
                if "skipped" in result:
                    n_skipped += 1
                    reason = result["skipped"]
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    print(f"  [{i + 1}/{len(items)}] SKIP {item['item_id']}: {reason}", flush=True)
                    continue
                for row in result["rows"]:
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                if logits_file is not None:
                    for entry in result["logits"]:
                        logits_file.write(json.dumps(entry, separators=(",", ":")) + "\n")
                n_scored += 1
                if (i + 1) % 10 == 0 or (i + 1) == len(items):
                    print(f"  [{i + 1}/{len(items)}] scored (last: {result['score_s']:.2f}s, "
                          f"{result['n_frames']} frames)", flush=True)
    finally:
        if logits_file is not None:
            logits_file.close()

    print(f"[+] done: {n_scored} scored, {n_skipped} skipped {json.dumps(skip_reasons, sort_keys=True)}")
    print(f"[+] rows: {args.out}" + (f" | logits: {args.dump_logits}" if args.dump_logits else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
