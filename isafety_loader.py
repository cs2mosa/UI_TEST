from __future__ import annotations
"""
isafety_loader.py - iSafetyBench annotation loader for JEV calibration (design_v3 §7/C0).

Documented schema (official repo github.com/iSafetyBench/data; validated loudly here,
since the actual snapshot layout is only inspected on the dataset host machine):

  <dataset_dir>/annotations_hazard.json   - JSON list of {video_name, caption, gt_actions[]}
  <dataset_dir>/annotations_normal.json   - same record shape, routine actions
  <dataset_dir>/**/<video_name>           - video files, resolved by recursive filename index

Subset rule (design_v3 §5.3, applied against isafety_mapping.json):
  - positive for hazard h: the clip's gt_actions intersect h's mapped categories;
  - clean negative for all h: tags intersect only routine_categories (the normal file);
  - excluded (strict reading): a hazard-file clip tagged with ANY unmapped hazard
    category - the loader also reports how many such clips would have been kept under
    the loose reading (>=1 mapped action), for the developer's mapping review;
  - rows for unmappable hazard-on-item pairs are omitted by the harness, never
    assumed negative (labels carry None here).
Any gt_action outside the mapping's three lists aborts (design_v3 §9: silent partial
mapping is forbidden).
"""

HAZARD_ANNOTATIONS = "annotations_hazard.json"
NORMAL_ANNOTATIONS = "annotations_normal.json"

CalibrationItem = dict  # {item_id, video_path, hazard_labels: dict[hazard, "pos"|"neg"|None], tags, camera}


def _load_annotation_file(path: str) -> list[dict]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of records, got {type(data).__name__}")
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise ValueError(f"{path}: record {i} is not an object")
        missing = {"video_name", "gt_actions"} - set(rec)
        if missing:
            raise ValueError(f"{path}: record {i} missing keys {sorted(missing)} (schema: video_name, caption, gt_actions)")
        if not isinstance(rec["video_name"], str) or not rec["video_name"]:
            raise ValueError(f"{path}: record {i} has empty/invalid video_name")
        if not isinstance(rec["gt_actions"], list) or any(not isinstance(t, str) for t in rec["gt_actions"]):
            raise ValueError(f"{path}: record {i} gt_actions must be a list of strings")
    return data


def _video_index(dataset_dir: str) -> dict[str, str]:
    import os

    index: dict[str, str] = {}
    for root, _dirs, files in os.walk(dataset_dir):
        for name in files:
            if name not in index:
                index[name] = os.path.join(root, name)
    return index


def load_items(dataset_dir: str, mapping: dict) -> list[CalibrationItem]:
    """
    Returns one CalibrationItem per usable annotated clip:
      {item_id, video_path, hazard_labels: {hazard: "pos"|"neg"|None}, tags, camera}
    Raises ValueError on schema/mapping violations (the harness maps that to exit 2).
    """
    import os

    mapped: dict[str, set[str]] = {h: set(v) for h, v in mapping["hazards"].items()}
    mapped_all: set[str] = set().union(*mapped.values()) if mapped else set()
    unmapped: set[str] = set(mapping["unmapped_hazard_categories"])
    routine: set[str] = set(mapping["routine_categories"])
    hazards = list(mapped.keys())

    known = mapped_all | unmapped | routine
    overlap = (mapped_all & unmapped) | (mapped_all & routine) | (unmapped & routine)
    if overlap:
        raise ValueError(f"mapping lists a category in more than one section: {sorted(overlap)}")

    hazard_path = os.path.join(dataset_dir, HAZARD_ANNOTATIONS)
    normal_path = os.path.join(dataset_dir, NORMAL_ANNOTATIONS)
    for path in (hazard_path, normal_path):
        if not os.path.isfile(path):
            raise ValueError(
                f"missing {os.path.basename(path)} in {dataset_dir} (documented iSafetyBench layout; "
                "inspect the actual snapshot and adjust this loader if the layout differs)"
            )

    hazard_records = _load_annotation_file(hazard_path)
    normal_records = _load_annotation_file(normal_path)

    unknown_tags: dict[str, int] = {}
    for rec in hazard_records + normal_records:
        for tag in rec["gt_actions"]:
            if tag not in known:
                unknown_tags[tag] = unknown_tags.get(tag, 0) + 1
    if unknown_tags:
        raise ValueError(
            f"mapping is missing {len(unknown_tags)} category(-ies) present in the data "
            f"(silent partial mapping is forbidden): "
            + ", ".join(f"{t!r} x{n}" for t, n in sorted(unknown_tags.items()))
        )

    videos = _video_index(dataset_dir)
    if not videos:
        raise ValueError(f"no video files found under {dataset_dir}")

    items: list[CalibrationItem] = []
    n_no_video, n_excluded_strict, n_excluded_loose_kept = 0, 0, 0
    for rec in hazard_records:
        tags = list(dict.fromkeys(rec["gt_actions"]))  # dedupe, keep order
        if any(t in unmapped for t in tags):
            n_excluded_strict += 1
            if any(t in mapped_all for t in tags):
                n_excluded_loose_kept += 1  # would be positive for some hazard under the loose reading
            continue
        hit = [h for h in hazards if any(t in mapped[h] for t in tags)]
        if not hit:
            continue  # hazard clip with no mapped action: neither positive nor clean negative
        video_path = videos.get(rec["video_name"])
        if video_path is None:
            n_no_video += 1
        labels = {h: ("pos" if h in hit else None) for h in hazards}
        items.append({
            "item_id": rec["video_name"],
            "video_path": video_path,
            "hazard_labels": labels,
            "tags": tags,
            "camera": None,
        })

    for rec in normal_records:
        tags = list(dict.fromkeys(rec["gt_actions"]))
        video_path = videos.get(rec["video_name"])
        if video_path is None:
            n_no_video += 1
        labels = {h: "neg" for h in hazards}
        items.append({
            "item_id": rec["video_name"],
            "video_path": video_path,
            "hazard_labels": labels,
            "tags": tags,
            "camera": None,
        })

    print(
        f"[isafety_loader] hazard-file clips kept: {sum(1 for i in items if any(v == 'pos' for v in i['hazard_labels'].values()))}; "
        f"normal-file clips: {len(normal_records)}; excluded (strict unmapped-tag rule): {n_excluded_strict} "
        f"(of which {n_excluded_loose_kept} would be kept under the loose reading); "
        f"video file missing: {n_no_video}",
        flush=True,
    )
    return items
