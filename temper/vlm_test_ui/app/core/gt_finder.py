from __future__ import annotations
import os
from typing import Optional


def find_ground_truth_file(media_path: str, project_dir: str) -> Optional[str]:
    """
    Auto-discovers ground truth annotations for a media file based on Design Doc §7.
    Search order:
    1. Sidecar next to media file: <stem>.gt.csv or <stem>.gt.json
    2. <project_dir>/ground_truth/<stem>.csv or .json
    3. Vicinity (+/- 1 directory level)
    """
    if not media_path:
        return None

    media_dir = os.path.dirname(os.path.abspath(media_path))
    stem = os.path.splitext(os.path.basename(media_path))[0]

    # 1. Sidecar next to media file
    candidates = [
        os.path.join(media_dir, f"{stem}.gt.csv"),
        os.path.join(media_dir, f"{stem}.gt.json"),
        os.path.join(media_dir, f"{stem}.csv"),
        os.path.join(media_dir, f"{stem}.json"),
    ]

    # 2. Project ground_truth folder
    if project_dir:
        gt_dir = os.path.join(project_dir, "ground_truth")
        candidates.extend([
            os.path.join(gt_dir, f"{stem}.csv"),
            os.path.join(gt_dir, f"{stem}.json"),
            os.path.join(gt_dir, f"{stem}.gt.csv"),
            os.path.join(gt_dir, f"{stem}.gt.json"),
        ])

    # 3. Vicinity check (+1 directory up or subfolder 'ground_truth')
    parent_dir = os.path.dirname(media_dir)
    candidates.extend([
        os.path.join(parent_dir, f"{stem}.gt.csv"),
        os.path.join(parent_dir, "ground_truth", f"{stem}.csv"),
        os.path.join(media_dir, "ground_truth", f"{stem}.csv"),
    ])

    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    return None
