#!/usr/bin/env python3
"""
Warehouse hazard monitor - simplest two-stage pipeline.

    stream -> sampled frames -> sliding window -> decision model (OneJev)
           -> threshold + debounce -> clip -> VLM confirmation -> commentary

Setup
-----
  pip install opencv-python numpy
  # decision model (OneJev, Apache-2.0) - start the server separately:
  pip install "qev[torch] @ git+https://github.com/OmniJev/OneJev.git"
  qev serve --model OmniJev/OneJev-4B --port 8000
  # VLM backend (pick one):
  pip install anthropic        # --vlm anthropic  (needs ANTHROPIC_API_KEY)
  pip install openai           # --vlm openai     (vLLM/Ollama/any OpenAI-compatible server)

Try it with no models at all (synthetic video + mock backends):
  python hazard_monitor.py --make-demo demo.mp4
  python hazard_monitor.py --source demo.mp4 --decider mock --vlm mock --no-pace

Real run:
  python hazard_monitor.py --source rtsp://cam/stream --camera dock-3 \
      --decider onejev --decider-url http://localhost:8000 --vlm anthropic

Outputs (in --out-dir): scores.jsonl (every window's probabilities - use it to tune
thresholds), events.jsonl (every triggered event + VLM verdict), clips/*.mp4.
"""
import argparse
import base64
import collections
import concurrent.futures as cf
import json
import os
import re
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Hazard question panel (fixed, so scores are comparable across time/cameras)  #
# Each is a statement; the decision model returns P(statement is true).        #
# --------------------------------------------------------------------------- #
HAZARDS = {
    "pedestrian_vehicle": "A person is in danger of being struck by a forklift or other moving vehicle",
    "no_ppe": "A person is working without required PPE (hi-vis vest or hard hat)",
    "blocked_exit": "An aisle, fire exit or emergency equipment is blocked",
    "unsafe_load": "A load is unstable, overhanging or about to fall",
    "fall_or_spill": "A person has fallen, or there is a spill or obstruction on the floor",
}

VLM_PROMPT = """You are a warehouse safety inspector reviewing frames from a CCTV clip.
The frames are in time order and span about {span:.1f} seconds.
An automated screen flagged possible hazards: {flags}.

Look carefully. Report only what you can actually see; if you cannot tell, say so.
Respond with ONLY a JSON object, no other text:
{{"confirmed": true or false,
  "commentary": "1-3 plain-language sentences describing what is happening",
  "findings": [{{"description": "...", "hazard_type": "...", "severity": "none|low|medium|high",
                 "safe": true or false, "confidence": "low|medium|high"}}]}}"""


@dataclass
class Config:
    source: str
    camera: str = "cam-1"
    sample_fps: float = 4.0       # frames/sec fed to the pipeline
    window_s: float = 3.0         # length of the clip shown to the decision model
    stride_s: float = 1.5         # how often a window is evaluated
    preroll_s: float = 3.0        # extra lead-in kept for the VLM clip
    threshold: float = 0.35       # keep LOW: a missed hazard costs more than a VLM call
    min_consecutive: int = 1      # windows over threshold required before firing
    cooldown_s: float = 10.0      # per-hazard refractory period
    max_side: int = 640           # downscale frames so longest side is this many px
    vlm_max_frames: int = 8       # frames sent to the VLM
    out_dir: str = "out"
    pace: bool = True             # sleep to real time when reading a file


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def encode_jpeg(frame, max_side, quality=80):
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def data_uri(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def pick_evenly(items, n):
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def append_jsonl(path, obj, lock=threading.Lock()):
    with lock, open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


# --------------------------------------------------------------------------- #
# Stage 1: decision models. Interface: (jpeg_frames, fps, camera) -> {hazard: p}#
# --------------------------------------------------------------------------- #
class OneJevDecider:
    """OneJev served via `qev serve`. One request = one video window + all questions."""

    def __init__(self, url):
        from qev import Client, Noul  # pip install qev from the OneJev repo
        self.client = Client(url)
        self.questions = {k: Noul(v) for k, v in HAZARDS.items()}

    def __call__(self, jpegs, fps, camera):
        r = self.client.system_one(
            state={"task": "Warehouse CCTV safety monitoring. Judge the clip.",
                   "camera": camera, "clip": "<video:1>"},
            media=[{"type": "video", "frames": [data_uri(j) for j in jpegs], "fps": fps}],
            questions=self.questions,
        )
        return {k: float(r.answers[k].noul) for k in HAZARDS}  # P(yes)


class MockDecider:
    """Pipeline test only: pseudo-probability from frame-to-frame motion."""

    def __call__(self, jpegs, fps, camera):
        g = [cv2.cvtColor(cv2.imdecode(np.frombuffer(j, np.uint8), 1), cv2.COLOR_BGR2GRAY)
             .astype(np.float32) for j in jpegs]
        motion = float(np.mean([np.abs(a - b).mean() for a, b in zip(g, g[1:])])) if len(g) > 1 else 0.0
        p = {k: 0.02 for k in HAZARDS}
        p["pedestrian_vehicle"] = min(1.0, motion / 0.4)
        return p


# --------------------------------------------------------------------------- #
# Stage 2: VLM confirmation. Interface: (jpeg_frames, prompt) -> raw text      #
# --------------------------------------------------------------------------- #
class AnthropicVLM:
    def __init__(self, model):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model

    def __call__(self, jpegs, prompt):
        content = []
        for i, j in enumerate(jpegs, 1):
            content.append({"type": "text", "text": f"Frame {i}/{len(jpegs)}"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(j).decode()}})
        content.append({"type": "text", "text": prompt})
        msg = self.client.messages.create(
            model=self.model, max_tokens=1000,
            messages=[{"role": "user", "content": content}])
        return msg.content[0].text


class OpenAICompatVLM:
    """Any OpenAI-compatible server (vLLM, Ollama, ...) hosting a vision model."""

    def __init__(self, base_url, model, api_key="none"):
        from openai import OpenAI
        self.client, self.model = OpenAI(base_url=base_url, api_key=api_key), model

    def __call__(self, jpegs, prompt):
        content = [{"type": "image_url", "image_url": {"url": data_uri(j)}} for j in jpegs]
        content.append({"type": "text", "text": prompt})
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=1000,
            messages=[{"role": "user", "content": content}])
        return r.choices[0].message.content


class MockVLM:
    def __call__(self, jpegs, prompt):
        return json.dumps({"confirmed": True,
                           "commentary": f"[mock] reviewed {len(jpegs)} frames.",
                           "findings": [{"description": "mock finding", "hazard_type": "pedestrian_vehicle",
                                         "severity": "medium", "safe": False, "confidence": "low"}]})


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
class Monitor:
    def __init__(self, cfg: Config, decider, vlm):
        self.cfg, self.decider, self.vlm = cfg, decider, vlm
        self.buf = collections.deque()          # (ts, jpeg)
        self.lock = threading.Lock()
        self.busy = threading.Event()           # a decision call is in flight
        self.decide_pool = cf.ThreadPoolExecutor(1)
        self.vlm_pool = cf.ThreadPoolExecutor(1)
        self.last_tick = -1e9
        self.streak = collections.Counter()
        self.last_fire = {}
        self.dropped = 0
        os.makedirs(os.path.join(cfg.out_dir, "clips"), exist_ok=True)
        self.scores_path = os.path.join(cfg.out_dir, "scores.jsonl")
        self.events_path = os.path.join(cfg.out_dir, "events.jsonl")

    # called from the grabber loop - must stay fast
    def push(self, ts, frame):
        jpg = encode_jpeg(frame, self.cfg.max_side)
        with self.lock:
            self.buf.append((ts, jpg))
            horizon = ts - (self.cfg.window_s + self.cfg.preroll_s) - 1.0
            while self.buf and self.buf[0][0] < horizon:
                self.buf.popleft()
        self._maybe_tick(ts)

    def _maybe_tick(self, ts):
        c = self.cfg
        if ts - self.last_tick < c.stride_s:
            return
        with self.lock:
            clip = list(self.buf)               # snapshot (refs to immutable bytes)
        if not clip or ts - clip[0][0] < c.window_s * 0.95:
            return                              # not enough history yet
        self.last_tick = ts
        if self.busy.is_set():
            if self.cfg.pace:                   # live: never block the stream, skip this window
                self.dropped += 1
                return
            while self.busy.is_set():           # offline replay (--no-pace): wait so every window is scored
                time.sleep(0.005)
        window = [f for f in clip if f[0] >= ts - c.window_s]
        self.busy.set()
        self.decide_pool.submit(self._decide, ts, clip, window)

    def _decide(self, ts, clip, window):
        try:
            probs = self.decider([j for _, j in window], self.cfg.sample_fps, self.cfg.camera)
        except Exception as e:                  # decision failures must be visible, not silent
            append_jsonl(self.events_path, {"ts": ts, "status": "decider_error", "error": repr(e)})
            print(f"[{ts:7.1f}s] DECIDER ERROR: {e!r}")
            return
        finally:
            self.busy.clear()
        append_jsonl(self.scores_path, {"ts": round(ts, 2), "probs": {k: round(v, 4) for k, v in probs.items()}})
        fired = self._fired(ts, probs)
        if fired:
            self.vlm_pool.submit(self._confirm, ts, clip, probs, fired)

    def _fired(self, ts, probs):
        c, fired = self.cfg, []
        for h, p in probs.items():
            self.streak[h] = self.streak[h] + 1 if p >= c.threshold else 0
            if self.streak[h] >= c.min_consecutive and ts - self.last_fire.get(h, -1e9) >= c.cooldown_s:
                fired.append(h)
        for h in fired:
            self.last_fire[h] = ts
        return fired

    def _save_clip(self, ts, clip):
        path = os.path.join(self.cfg.out_dir, "clips", f"{self.cfg.camera}_{ts:010.1f}.mp4")
        first = cv2.imdecode(np.frombuffer(clip[0][1], np.uint8), 1)
        h, w = first.shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.cfg.sample_fps, (w, h))
        for _, j in clip:
            vw.write(cv2.imdecode(np.frombuffer(j, np.uint8), 1))
        vw.release()
        return path

    def _confirm(self, ts, clip, probs, fired):
        clip_path = self._save_clip(ts, clip)
        frames = pick_evenly([j for _, j in clip], self.cfg.vlm_max_frames)
        flags = "; ".join(f"{h} (p={probs[h]:.2f}) - {HAZARDS[h]}" for h in fired)
        event = {"ts": round(ts, 2), "camera": self.cfg.camera, "fired": fired,
                 "probs": {k: round(v, 4) for k, v in probs.items()}, "clip": clip_path}
        try:
            raw = self.vlm(frames, VLM_PROMPT.format(span=clip[-1][0] - clip[0][0], flags=flags))
            result = parse_json(raw)
            if result is None:
                event.update(status="unverified", reason="vlm_unparseable", vlm_raw=(raw or "")[:1000])
            else:
                event.update(status="confirmed" if result.get("confirmed") else "dismissed", vlm=result)
        except Exception as e:
            # Fail safe: if the VLM is down, the alert is NOT dismissed - route it for human review.
            event.update(status="unverified", reason="vlm_error", error=repr(e))
        append_jsonl(self.events_path, event)
        tag = event["status"].upper()
        text = (event.get("vlm") or {}).get("commentary", event.get("reason", ""))
        print(f"[{ts:7.1f}s] {tag:10s} {','.join(fired)} :: {text}")

    def close(self):
        self.decide_pool.shutdown(wait=True)
        self.vlm_pool.shutdown(wait=True)
        if self.dropped:
            print(f"note: {self.dropped} windows skipped because the decision model was busy "
                  f"- raise --stride-s, lower --sample-fps, or use a smaller/faster model")


# --------------------------------------------------------------------------- #
# Stream loop                                                                  #
# --------------------------------------------------------------------------- #
def run(cfg: Config, monitor: Monitor):
    is_file = os.path.isfile(cfg.source)
    src = int(cfg.source) if cfg.source.isdigit() else cfg.source
    cap = cv2.VideoCapture(src)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    start, idx, next_sample = time.monotonic(), 0, 0.0
    try:
        while True:
            if not cap.grab():                  # grab() is cheap; only decode sampled frames
                if is_file:
                    break
                cap.release(); time.sleep(2.0); cap = cv2.VideoCapture(src)   # live: reconnect
                continue
            ts = idx / src_fps if (is_file and src_fps > 0) else time.monotonic() - start
            idx += 1
            if is_file and cfg.pace:
                time.sleep(max(0.0, start + ts - time.monotonic()))
            if ts < next_sample:
                continue
            next_sample = ts + 1.0 / cfg.sample_fps
            ok, frame = cap.retrieve()
            if ok:
                monitor.push(ts, frame)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        monitor.close()


def make_demo(path, seconds=24, fps=20, w=640, h=360):
    """Synthetic clip: a static 'person' (blue) and a 'forklift' (red) that rushes toward them at t=10s."""
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(seconds * fps):
        t = i / fps
        img = np.full((h, w, 3), 90, np.uint8)
        cv2.rectangle(img, (400, 200), (430, 260), (200, 80, 0), -1)
        x = 40 if t < 10 else min(380, 40 + int((t - 10) * 120))
        cv2.rectangle(img, (x, 190), (x + 70, 250), (0, 0, 200), -1)
        vw.write(img)
    vw.release()
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-demo", metavar="PATH")
    ap.add_argument("--source", help="RTSP/HTTP URL, file path, or webcam index")
    ap.add_argument("--camera", default="cam-1")
    ap.add_argument("--decider", choices=["onejev", "mock"], default="onejev")
    ap.add_argument("--decider-url", default="http://localhost:8000")
    ap.add_argument("--vlm", choices=["anthropic", "openai", "mock"], default="anthropic")
    ap.add_argument("--vlm-model", default=None, help="model name for the VLM backend")
    ap.add_argument("--vlm-url", default="http://localhost:8001/v1", help="for --vlm openai")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--no-pace", action="store_true", help="read files as fast as possible")
    for name, typ in [("sample_fps", float), ("window_s", float), ("stride_s", float), ("preroll_s", float),
                      ("threshold", float), ("cooldown_s", float), ("min_consecutive", int),
                      ("max_side", int), ("vlm_max_frames", int)]:
        ap.add_argument("--" + name.replace("_", "-"), type=typ, default=getattr(Config, name))
    a = ap.parse_args()

    if a.make_demo:
        return make_demo(a.make_demo)
    if not a.source:
        ap.error("--source is required")

    cfg = Config(source=a.source, camera=a.camera, sample_fps=a.sample_fps, window_s=a.window_s,
                 stride_s=a.stride_s, preroll_s=a.preroll_s, threshold=a.threshold,
                 min_consecutive=a.min_consecutive, cooldown_s=a.cooldown_s, max_side=a.max_side,
                 vlm_max_frames=a.vlm_max_frames, out_dir=a.out_dir, pace=not a.no_pace)

    decider = MockDecider() if a.decider == "mock" else OneJevDecider(a.decider_url)
    if a.vlm == "mock":
        vlm = MockVLM()
    elif a.vlm == "openai":
        vlm = OpenAICompatVLM(a.vlm_url, a.vlm_model or "Qwen/Qwen3-VL-8B-Instruct")
    else:
        vlm = AnthropicVLM(a.vlm_model or os.environ.get("VLM_MODEL", "claude-sonnet-5-5"))
    run(cfg, Monitor(cfg, decider, vlm))


if __name__ == "__main__":
    main()
