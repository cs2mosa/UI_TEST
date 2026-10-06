from __future__ import annotations
"""
frame_sampler.py - Frame sampling utilities (numpy + cv2 only).

Reproduces BatchWorker._sample_frames_for_window logic exactly (F-2)
as pure functions, plus resize and JPEG thumbnail helpers.
"""
import numpy as np
import cv2


def target_frame_count(start_ms: int, end_ms: int, fps: float) -> int:
    """Identical to the baseline formula: max(2, round((end-start)/1000 * fps))."""
    duration_s = (end_ms - start_ms) / 1000.0
    return max(2, int(round(duration_s * fps)))


def sample_timestamps(start_ms: int, end_ms: int, fps: float) -> list[int]:
    """Evenly-spaced timestamps (inclusive endpoints) at the given fps."""
    n = target_frame_count(start_ms, end_ms, fps)
    return np.linspace(start_ms, end_ms, n, dtype=int).tolist()


def read_frames_at(
    cap: cv2.VideoCapture,
    timestamps_ms: list[int],
    rgb: bool = True,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Reproduce F-2 exactly:
    - Per timestamp: cap.set(POS_MSEC, ts), cap.read(); stop at first failed read.
    - Record int(cap.get(POS_MSEC)) as the actual timestamp after each read.
    - If no frame was read and timestamps_ms is non-empty: one fallback read at timestamps_ms[0].
    - If timestamps_ms is empty: return ([], []).
    - rgb=True: convert BGR -> RGB; rgb=False: return frame as-is (BGR).
    """
    if not timestamps_ms:
        return [], []

    frames: list[np.ndarray] = []
    actual_ts: list[int] = []

    for ts in timestamps_ms:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts)
        ret, frame = cap.read()
        if not ret:
            break
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        actual_ts.append(int(cap.get(cv2.CAP_PROP_POS_MSEC)))
        frames.append(frame)

    if not frames:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamps_ms[0])
        ret, frame = cap.read()
        if ret:
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            actual_ts.append(int(cap.get(cv2.CAP_PROP_POS_MSEC)))
            frames.append(frame)

    return frames, actual_ts


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """
    Scale img so that max(h, w) <= max_side using INTER_AREA.
    Returns img unchanged if it already fits.
    Output dimensions are at least 1 px on each side.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_jpeg(img_bgr: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame as JPEG bytes. Raises RuntimeError on failure."""
    try:
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    except cv2.error as exc:
        raise RuntimeError(f"cv2.imencode failed: {exc}") from exc
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def make_thumb(img_bgr: np.ndarray, max_side: int = 160, quality: int = 70) -> bytes:
    """Resize then JPEG-encode a BGR frame."""
    small = resize_max_side(img_bgr, max_side)
    return encode_jpeg(small, quality=quality)