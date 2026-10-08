# JEV-Gated Frame-Rate Routing for UI_TEST — Design Document

**Version:** 2.0 · **Date:** 5 Oct 2026 · **Repo baseline:** `UI_TEST` @ `fedfb07` ("second try"), project directory updated by the developer, for the rest of the project, you dont need git status for reviewing the former commits, the codebase is in its newest update up until now. (register D-2)
**Supersedes:** v1.0 ("JEV Integration into UI_TEST"). v1.0's side-by-side policy (its Option A) is **dropped**; this document implements a JEV **router** in front of HERMES.
**Status:** Decision Register (§14) confirmed in conversation with the developer on 5 Oct 2026. Implementation proceeds phase-by-phase per §13, stopping after every phase (D-15).
**Package note:** this document is bundled inside the `jev-integration` skill package. Every source file it references under `references/sources/` travels with the package (`hazard_monitor.py`, `qev/*.py` — verbatim, Apache-2.0); the package is otherwise self-contained: repo, venv and model-cache locations are pinned as environment facts (D-2, D-19, D-22).

---

## 0. How to read this document

### 0.1 Tags
Every statement carries one of four tags. The tag decides what an agent may do with it.

| Tag | Meaning | Agent may |
| --- | --- | --- |
| **[GIVEN]** | Stated by the developer. Binding. | Act on it. |
| **[VERIFIED]** | Observed by reading the code or data (method noted). | Rely on it; re-check if the code differs from `fedfb07`. |
| **[PROPOSED]** | Designer's choice. Binding **only after** §14 is confirmed. | Act on it after confirmation. |
| **[OPEN]** | Unknown. Needs developer input. | Not act on anything that depends on it. Ask. |

"MUST" / "MUST NOT" are binding. "e.g." is illustrative, never binding.

### 0.2 Agent contract (absolute rules)
1. Do only what the Change Budget (§7.2) and the current phase (§13) allow. Anything else: **stop and ask**.
2. **Never assume.** If this document is silent, ambiguous, self-contradicting, or reality differs from a [VERIFIED] fact: stop and report `QUESTION: <section> · <what you saw> · <options>`. Do not pick an option yourself.
3. **Never write OneJev client calls, JEV question texts, or any `hazard_monitor.py` logic from memory.** Copy only from the supplied sources: `hazard_monitor.py` (v0.1 reference, supplied by the developer) and the `qev` package sources copied verbatim to `references/sources/qev/ (bundled in this package)` (register D-16, D-17).
4. **Never change** a default value, name, signature, file name, format string or expected test value written in this document. If a test looks wrong, report it; do not edit the expected value.
5. Do not run GPU, model, or network jobs. Tests marked **DEV** are run by the developer.
6. Write each test before the code it covers. A phase is complete only with the evidence in §12.3.
7. No refactoring, reformatting, renaming, reordering, "cleanup", or dependency changes in existing files. Additions only, exactly as §7.2 lists them.
8. Do not add features that are not in this document (see §1.4). "While I'm here" changes are forbidden.

---

## 1. Purpose and scope

### 1.1 Goal
Put the **JEV** hazard scorer in front of the existing HERMES VLM. For every video chunk JEV decides **how much visual data HERMES receives** (full or reduced frame rate), so that HERMES spends less time on chunks that look clear, keeps full detail on chunks that look risky, and raises its level of detail **before** a violation fully develops. **[GIVEN]** (C-9, C-10)

### 1.2 Developer constraints
| ID | Constraint | Tag |
| --- | --- | --- |
| C-1 | Integrate JEV into the UI_TEST codebase. | **[GIVEN]** |
| C-2 | The UI exists for **visual testing** of what the system saw and how it judged it. | **[GIVEN]** |
| C-3 | Input is a recorded video of 5–60 min, **simulated** as a livestream. No real RTSP/camera work. | **[GIVEN]** — this document implements **Stage 1** (the existing offline *Process* phase, §1.3); the live-stream simulation is **Stage 2**, out of scope here (D-1). |
| C-4 | Only HERMES's KV-caching strategy is used. InfiniPot-V is not part of the integration work. | **[GIVEN]** |
| C-5 | **No timeout feature, ever** (precise meaning in §10.1). | **[GIVEN]** |
| C-6 | Minimal code changes; the existing system must not break. | **[GIVEN]** |
| C-7 | The implementing agent assumes nothing and returns to the developer for permission. | **[GIVEN]** |
| C-8 | Final step: change `compare_latency.py` from comparing 2 systems to comparing 3. | **[GIVEN]** |
| C-9 | JEV is added on top of the VLM to route the relevant safety-violation frames to it, so the system detects an ongoing violation faster and flags rising risk earlier. | **[GIVEN]** |
| C-10 | When JEV judges a chunk clear, the VLM still receives it, at a **reduced frame rate**. No chunk is skipped. | **[GIVEN]** |
| C-11 | JEV runs **in-process** via the `transformers` library on **OneJev-0.8B**. No HTTP server, no ports, no `qev` client: the code must be self-contained because it will later serve as an `ask_qwen` function inside an NVIDIA DeepStream pipeline. | **[GIVEN]** (register D-16) |

### 1.3 Stages
- **Stage 1 (this document):** the existing *Process → Inspect* flow. A new `GatedBatchWorker` replaces `BatchWorker` **only when a JEV decider is supplied**. For each chunk it runs JEV, routes, samples frames at the routed rate, and calls the existing `VLMAdapter`. JEV and HERMES run **sequentially** inside one worker thread.
- **Stage 2 (not in this document):** paced live-stream simulation, concurrent JEV/HERMES threads, lag accounting. The pure-logic modules written here (`jev_gate`, `frame_sampler`, `jev_adapter`) MUST NOT depend on Qt so Stage 2 can reuse them.

### 1.4 Non-goals (an agent MUST NOT implement any of these)
- Live-stream simulation, pacing, frame ring buffers, concurrent JEV/HERMES threads.
- Skipping a chunk, skipping HERMES generation, or any change to HERMES internals (`vlm/hermes_*.py`, `vlm/vision_common.py`).
- The v0.1 `hazard_monitor.py` trigger logic (`min_consecutive`, cooldown). Not used (D-7).
- Latent-condition tracking, zones, notifications, model training.
- A JEV overlay on `VideoView`, timeline lanes, or new toolbar buttons (D-9).
- Fixing existing defects (§2.4). They are reported, not fixed.

### 1.5 What "faster" means and how it is measured (no claims; measurements only)
| Claim | Measure | Where |
| --- | --- | --- |
| HERMES does less work on clear chunks | `ingest_time_s` and `total_time_s` at reduced vs full fps on the same chunks | G0 (§13) |
| Later chunks are reached sooner | `wall_done_s` per chunk (§6.2) in the gated run vs a baseline run | DEV-3 |
| JEV flags risk before HERMES reports it | In each chunk's JEV ticks, compare tick results with the HERMES event of the same chunk | UI + JSONL |
| Rising risk is flagged early | Chunks with reason `rising` / `above_watch` that precede the first `above_full` chunk | G1 analysis (Appendix B) |
| No violation chunk is down-rated | Count of GT-positive chunks routed `REDUCED` (target 0) | G1 (Appendix B) |

**Latency neutrality of the integration itself (developer requirement):** the only added per-chunk work is JEV tick evaluation (decode + resize + decider forward passes). Every added millisecond is measured and recorded (`jev_time_s` per tick in §5.2, summed per chunk in §6.2) alongside `vlm_time_s` and `wall_done_s`, so the net benefit is always computable as Σ(VLM time saved on REDUCED chunks) − Σ(`jev_time_s`). No hidden work is introduced anywhere else: no extra preprocessing of VLM frames, no changes to the HERMES path, no copies of the chunk frames. Tick frames and chunk frames are decoded in two separate passes over the file — deliberate, to keep tick-window sampling independent of chunk sampling; the duplication costs well under a second per chunk against ~100 s of HERMES ingest (F-9) and is itself included in `jev_time_s`.

### 1.6 Baseline verification (5 Oct 2026)
The [VERIFIED] facts below were re-checked by reading the supplied project directory (`UI_TEST` @ `fedfb07`, clean tree): F-1..F-13 and F-15 confirmed by reading `batch_worker.py`, `types.py`, `config.py`, `vlm_adapter.py`, `main.py`, `main_window.py`, `exporter.py`, `session_model.py`, `chunker.py`, `hermes_runner.py`, `compare_latency.py`, `requirements.txt`; F-9 confirmed by reading `hermes_benchmark_telemetry.json`; F-16 carried from v1.0 (re-run in Phase 0). F-17 is **resolved** (see its row).

---

## 2. Baseline facts (the code as it is)

Method key: **R** = read the file in this revision; **D** = read the tracked data file; **C** = carried from v1.0 (not re-checked here; the agent re-checks them in Phase 0 via R-1..R-3).

### 2.1 The application **[VERIFIED: R]**
PySide6 desktop harness, two phases. **Process:** `BatchWorker` cuts the file into chunks with `compute_chunk_windows` (default 5 s chunk, 1 s overlap, 4 fps), calls `VLMAdapter.process_chunk` per chunk and emits `VLMEvent` through `event_ready`. **Inspect:** `PlayerWorker` plays the file, `SessionModel` supplies the event at the playhead, widgets show feed/timeline/summary, GT matching, verdicts, JSON/CSV export. Runners: `MockVLM`, `Qwen3VLRunner`, `HermesStreamingRunner` (stateful; `BatchWorker` detects it with `hasattr(vlm, "start_session")`).

### 2.2 Facts that shape the design
| ID | Fact | Method |
| --- | --- | --- |
| F-1 | `VLMAdapter` has no timeout, although its docstring mentions timeouts. | R |
| F-2 | `BatchWorker._sample_frames_for_window`: `n = max(2, int(round(duration_s * vlm_fps)))`, `np.linspace(start_ms, end_ms, n, dtype=int)`, per timestamp `cap.set(CAP_PROP_POS_MSEC, ts)` + `cap.read()`, BGR→RGB, stop at the first failed read; if no frame was read, one fallback read at `start_ms`. Frames stay at source resolution. Timestamps are `int(cap.get(CAP_PROP_POS_MSEC))` after each read. | R |
| F-3 | `BatchWorker.run`: duration `int(round(frame_count / source_fps * 1000))`; windows from `compute_chunk_windows(..., include_partial=True)`; `event_id = idx + 1`; emits `progress(completed, total, eta_s)` after each chunk; `end_session()` on finish, cancel and exception. | R |
| F-4 | `BatchWorker.run` calls `start_session()` **before** checking the file opens and returns **without** `end_session()` on open failure or zero windows (existing defect, §2.4 #6). | R |
| F-5 | `SessionModel.add_event` replaces an event with the same `chunk_id`. `EventCard` renders any status other than `ok`/`invalid` as an error. **No new status string is introduced by this design.** | R |
| F-6 | `HermesStreamingRunner.__call__` per chunk: `encode_video_chunk` in sub-chunks of `encode_chunk_size` (8) with `frame_fps=chunk.fps`, then `predict_and_compress()`, then `question_answering(...)`. `question_answering` appends to `conv_history`, and `predict_and_compress` builds its pruning queries from `conv_history[-1]`. **Because Stage 1 still answers every chunk, `conv_history` evolves exactly as in baseline.** | R |
| F-7 | `preprocess_video_frames` pads an odd frame count by repeating the last frame (`n % 2 == 1`). Odd and small frame counts are therefore safe. | R |
| F-8 | HERMES temporal position IDs use the **engine-level** `sample_fps` (0.5, fixed at load) via `second_per_grid_t = 2.0 / sample_fps`, **not** `chunk.fps`. A reduced-rate chunk has fewer temporal slots, each slot spanning more video time than in a full-rate chunk, while positions advance by the same step. Effect on answer quality is **unmeasured** (risk R-3, gate G0-quality). The engine MUST NOT be changed. | R |
| F-9 | Tracked telemetry (`hermes_benchmark_telemetry.json`, one run, 5 chunks of 20 frames, hardware unknown, peak VRAM 3.2–10.3 GB): ingest = 89–95 s of 96–125 s per chunk (75–93 %); generation = 6–12 s (6–10 %); compression = 0.4–18.7 s. **Frame count is therefore the lever; generation gating alone would save ≤ ~10 %.** | D |
| F-10 | `SessionExporter.export_json(..., extra_meta=None)` merges `extra_meta` into the `session` block (`header.update(extra_meta)`). `exporter.py` therefore needs **no change**. | R |
| F-11 | `MainWindow._start_processing` builds `BatchWorker(...)` and connects five signals (`event_ready`, `progress`, `finished`, `canceled`, `error_occurred`) before `start()`. `closeEvent` calls `batch_worker.cancel()` then `wait(1000)` (pre-existing). `MainWindow.__init__(vlm, model_label, config, parent=None)`. | R |
| F-12 | `main.py` parses args **before** creating `QApplication`; `--help` therefore exits without a GUI. | R |
| F-13 | `compare_latency.py`: workers run in clean subprocesses, evict the other tree from `sys.path`/`sys.modules`; `sample_video_chunks` **pre-resizes frames to `--max-dim` 448** (the app does not); master writes fixed-name JSON files into the current directory; `print_comparison_table(hermes, infinipot)` is 2-engine; `--engine` choices are `hermes`, `infinipot`. | R |
| F-14 | `hermes_benchmark_telemetry.json` and `infinipot_benchmark_telemetry.json` are tracked in git at the repo root and are overwritten by a default benchmark run. | R (`git ls-files`) |
| F-15 | Hard-coded Windows paths (`F:\URCA_PROJECTS\...`) exist in `config.py`, `hermes_runner.py`, `qwen3vl_runner.py`, `compare_latency.py`, several tests. `tests/test_pipeline.py::test_state_flow` hard-codes one and hangs headless (modal `QMessageBox`). | R / C |
| F-16 | Mock flow on `examples/synthetic_test.mp4` (12 s, 24 fps, 640×360): 3 chunks `[0–5000]`, `[4000–9000]`, `[8000–12000]` ms with 20, 20, 15 frames (the last read at 12000 ms fails). 11 of the 12 headless-collectable tests pass. | C |
| F-17 | **Resolved 5 Oct 2026 (register D-3, D-16, D-17).** `hazard_monitor.py` (v0.1 reference) was supplied by the developer; the OneJev scoring internals were obtained from the official `OmniJev/OneJev` repository and copied verbatim under `references/sources/qev/ (bundled in this package)` (Apache-2.0): `prompt.py` (question rendering, `qev-labels-v2`), `media.py` (video/frame handling), `mm_engine.py` + `engine.py` (slot-logit readout), `calibrate.py` (softmax), `answers.py` (probability assembly), `schema.py`. The five hazard names `pedestrian_vehicle`, `no_ppe`, `blocked_exit`, `unsafe_load`, `fall_or_spill` and their question texts come from `hazard_monitor.HAZARDS`. | Supplied + fetched |

### 2.3 Invariants (must hold after every phase)
| ID | Invariant | Check |
| --- | --- | --- |
| I-1 | With no JEV decider supplied (`--jev off`, the default), behaviour is identical to `fedfb07`: same worker class, same windows, same events, same exports. | R-1..R-3, WIN-1 |
| I-2 | No existing public signature, signal, dataclass field, or status string is changed or removed. Only the additions in §7.2 are made; `MainWindow.__init__` keeps its positional order (new parameters are appended **after** `parent`). | `git diff` review |
| I-3 | These files are **not modified**: `temper/**`, `vlm/**`, `app/core/vlm_adapter.py`, `app/core/chunker.py`, `app/workers/batch_worker.py`, `app/workers/player_worker.py`, `app/io/exporter.py`, `app/widgets/video_view.py`, `app/widgets/settings_bar.py`, `app/widgets/event_feed.py`, `app/widgets/event_card.py`, `app/widgets/chunk_timeline.py`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, and existing tests. | `git diff --stat` |
| I-4 | No timeout mechanism is introduced (§10.1). | T-1 |
| I-5 | No new dependency. `torch` and `transformers` are imported **lazily and only inside `OneJevDecider`**, so `--jev off` and `--jev mock` never import them. The `qev` client/server package is **not** used at all: scoring is in-process (C-11, D-16). | `git diff`, JEV-8 |
| I-6 | Every chunk produced by `compute_chunk_windows` is sent to the VLM exactly once, in order, regardless of JEV output. | GATE-1, GATE-4 |

### 2.4 Known existing defects — reported, not fixed
1. `test_pipeline.py` hard-coded Windows path and modal-dialog hang (F-15).
2. Hard-coded `F:` paths; `qwen3vl_runner.py` overwrites `HF_HOME` at import.
3. `VLMAdapter` docstring claims timeout handling that does not exist (F-1). Under C-5 it must stay that way.
4. With a stateful runner and `overlap_s > 0`, HERMES ingests the overlapping second twice. Kept (D-12).
5. The tracked telemetry JSON files are overwritten by a default benchmark run (F-14).
6. `BatchWorker` starts the HERMES session before checking the file opens and does not end it on open failure or zero windows (F-4). `GatedBatchWorker` does **not** inherit this behaviour (§7.1).

---

## 3. Glossary (normative)

| Term | Definition |
| --- | --- |
| **chunk** | One window from `compute_chunk_windows(duration_ms, chunk_s, overlap_s, include_partial=True)`; index `k`, interval `[s_k, e_k]` ms. |
| **tick** | One JEV evaluation. Identified by its **end time** `t` (ms, stream time). Ticks are produced by `compute_tick_times` (§5.1) and numbered `tick_id = 0, 1, 2…` in ascending `t`. |
| **tick window** | `(max(0, t − window_ms), t]`; frames are sampled at `jev_sample_fps` (§5.2). |
| **region of chunk k** | `(e_{k-1}, e_k]`, with `e_{-1} = −∞`. The ticks whose end time lies in the region are **assigned to chunk k** (§5.3). Every tick belongs to exactly one chunk. |
| **vec** | Per-hazard maximum probability over the ticks of a chunk's region: `vec[h] = max over assigned ticks of probs[h]`. |
| **pmax** | `max(vec.values())`. |
| **tier** | `FULL` (frames sampled at the UI's `vlm_fps`, exactly as baseline) or `REDUCED` (frames sampled at `route_reduced_fps`). |
| **reason** | One of `jev_incomplete`, `above_full`, `above_watch`, `rising`, `hold`, `clear` (§5.4). |
| **fail-open** | When JEV information is missing or invalid, the chunk is routed `FULL`. A missing JEV result is never treated as "safe". |
| **hold** | After an `above_full` chunk, the next `route_hold_chunks` chunks that would otherwise be `REDUCED` stay `FULL` with reason `hold`. |
| **decider** | A callable `decider(frames_rgb: list[np.ndarray], fps: float, t_ms: int) -> dict[str, float]` returning one probability in [0, 1] for every hazard name. Frames are RGB uint8 HxWx3 (the harness convention). In-process; no HTTP (C-11, D-16, D-17). |

---

## 4. Architecture (Stage 1)

```
              ┌────────────────────────── GatedBatchWorker (one QThread, sequential) ──────────────────────────┐
 video file → │ for each chunk k:                                                                               │
              │   1. JEV ticks assigned to chunk k  → evaluate_tick() each   (decode+resize+decider, §5.2)      │
              │   2. GateRouter.decide(ticks)       → tier FULL | REDUCED + reason  (pure, §5.4)                │
              │   3. sample frames at tier rate     → VideoChunk (fps = tier rate)  (§5.5)                      │
              │   4. VLMAdapter.process_chunk(...)  → VLMEvent  (unchanged HERMES path, answers EVERY chunk)    │
              │   5. emit GateRecord, then VLMEvent, then progress; append JSONL lines (§6.4)                   │
              └─────────────────────────────────────────────────────────────────────────────────────────────────┘
 MainWindow: SessionModel (+gate_records) · GatePanel (new) · EventFeed/Timeline/Summary (unchanged)
```
- **Why sequential:** JEV is expected to cost well under a second per tick (0.8B model, in-process) versus ~100 s per HERMES chunk (F-9), so serial execution is the simplest correct design; Stage 2 revisits concurrency.
- **Why no gaps:** every chunk reaches HERMES (I-6, C-10). HERMES's KV cache, `conv_history`, session lifetime and answer cadence stay as baseline (F-6). Only the **number of frames** in a chunk changes, which is the dominant cost (F-9).
- **Rising risk:** the `above_watch` and `rising` rules lift a chunk to `FULL` **before** a violation crosses `route_full_threshold`; `hold` keeps detail after a violation.

---

## 5. Normative logic (pure functions; no Qt, torch or network)

### 5.1 Tick schedule — `compute_tick_times(duration_ms, window_ms, stride_ms) -> list[int]`
1. If `window_ms < 1` or `stride_ms < 1` or `stride_ms > window_ms`: raise `ValueError`.
2. If `duration_ms <= 0`: return `[]`.
3. `ticks = [window_ms + j*stride_ms for j = 0,1,2,… while value <= duration_ms]`.
4. If `ticks` is empty **or** `ticks[-1] < duration_ms`: append `duration_ms`.
5. Return `ticks`. Result is strictly ascending, ends at `duration_ms`, first element is `min(window_ms, duration_ms)`.

Examples: `(12000,3000,1500)` → `[3000,4500,6000,7500,9000,10500,12000]`; `(4000,3000,1500)` → `[3000,4000]`; `(2000,3000,1500)` → `[2000]`; `(3000,3000,1500)` → `[3000]`.
`window_ms = int(round(jev_window_s*1000))`, `stride_ms = int(round(jev_stride_s*1000))` (properties of `GateConfig`).

### 5.2 Tick evaluation — `evaluate_tick(decider, cap, tick_id, t_ms, cfg) -> TickRecord`
Start `perf_counter` at entry; `jev_time_s` is the total elapsed at return (decode + resize + convert + decider).
1. `window_start_ms = max(0, t_ms - cfg.window_ms)`.
2. `timestamps = sample_timestamps(window_start_ms, t_ms, cfg.jev_sample_fps)`; `frames_bgr, _ = read_frames_at(cap, timestamps, rgb=False)`.
3. If no frames: return `TickRecord(status="error", error="no_frames", num_frames=0, probs=None, thumbs=[])`. **The decider is not called.**
4. Resize each frame with `resize_max_side(frame, cfg.jev_max_side)`; convert BGR→RGB → `frames_rgb` (list of np.ndarray). **No JPEG encoding:** the decider consumes frames directly (D-17).
5. Thumbnails: indices `sorted({0, n//2, n-1})` of the resized BGR frames → `make_thumb(frame)` (max side 160, quality 70).
6. Call `decider(frames_rgb, cfg.jev_sample_fps, t_ms)` inside `try/except Exception` (never `BaseException`). On exception: `status="error"`, `error=f"{type(e).__name__}: {e}"`, `probs=None`, `thumbs=[]`.
7. `probs = validate_probs(raw)` (§7.3). A `ValueError` → `status="error"`, `error=str(e)` (starts with `invalid_output:`), `probs=None`, `thumbs=[]`.
8. Success: `status="ok"`, `probs` = validated dict, `thumbs` as step 5, `error=None`.
No retry, no timeout, no caching.

### 5.3 Tick-to-chunk assignment — `assign_ticks_to_chunks(tick_times, windows) -> list[list[int]]`
- Returns one list per window, each holding **indices into `tick_times`** (ascending).
- Tick `i` with time `t` belongs to chunk `k` iff `e_{k-1} < t <= e_k`, where `e_k = windows[k][1]` and `e_{-1} = −∞`.
- A tick with `t > e_last` is assigned to the last chunk.
- Every tick index appears exactly once. A chunk may have an empty list. If `windows` is empty return `[]`.

### 5.4 Routing — `GateRouter`
State: `prev_vec: dict|None = None`, `hold_remaining: int = 0`. `decide(ticks)` is called once per chunk, in order, with the chunk's assigned `TickRecord`s. Comparison helper: `ge(x, thr) := x >= thr - EPS` with `EPS = 1e-9`. First matching rule wins:

| # | Condition | Tier | Reason |
| --- | --- | --- | --- |
| 1 | `ticks` is empty **or** any tick has `status != "ok"` | FULL | `jev_incomplete` |
| 2 | `ge(pmax, route_full_threshold)` | FULL | `above_full` |
| 3 | `ge(pmax, route_watch_threshold)` | FULL | `above_watch` |
| 4 | `prev_vec is not None` and some hazard `h` has `ge(vec[h] - prev_vec[h], route_rise_delta)` | FULL | `rising` |
| 5 | `hold_remaining > 0` | FULL | `hold` |
| 6 | otherwise | REDUCED | `clear` |

- `vec[h] = max(tick.probs[h] for tick in ticks)` for every `h` in `HAZARDS`; `pmax = max(vec.values())`. Computed only when rule 1 does not match; for rule 1, `vec=None`, `pmax=None`.
- An `ok` tick whose `probs` is `None` or lacks a `HAZARDS` key: raise `ValueError` (programming error; `evaluate_tick` prevents it).
- `rising_hazard` = the first hazard in `HAZARDS` order satisfying rule 4 (only when reason is `rising`; otherwise `None`).
- **State update after the decision:** `reason=="above_full"` → `hold_remaining = route_hold_chunks`; `reason=="hold"` → `hold_remaining -= 1`; any other reason → unchanged. Then `prev_vec = vec` (a copy) if `reason != "jev_incomplete"` else `None`.
- `RouteDecision.hold_remaining_after` = `hold_remaining` after the update.
- The router never mutates its inputs and has no randomness or clock access.

### 5.5 Frame sampling for the VLM chunk
`FULL` → fps = the UI's `vlm_fps` (unchanged baseline sampling). `REDUCED` → fps = `route_reduced_fps`. In both cases frames come from `read_frames_at(cap, sample_timestamps(start_ms, end_ms, fps), rgb=True)`, where
- `target_frame_count(s, e, fps) = max(2, int(round((e - s)/1000.0 * fps)))` (Python's `round`, deliberately identical to the baseline formula),
- `sample_timestamps(s, e, fps) = np.linspace(s, e, target_frame_count(s, e, fps), dtype=int).tolist()`,
- `read_frames_at` reproduces F-2 exactly: per timestamp `cap.set(CAP_PROP_POS_MSEC, ts)`, `cap.read()`, **stop at the first failed read**, record `int(cap.get(CAP_PROP_POS_MSEC))` as the timestamp; if no frame was read, one fallback read at `timestamps[0]` (if `timestamps` is empty return `([], [])`); `rgb=True` converts BGR→RGB, `rgb=False` returns the frame as read.
The resulting `VideoChunk.fps` is the tier fps. Frames are **not** resized (baseline behaviour; HERMES resizes internally, F-8/`vision_common.py`). `VideoChunk` is frozen; construct it once with the final lists.

Expected counts (`target_frame_count`): `(0,5000,4.0)=20`, `(0,5000,1.0)=5`, `(8000,12000,4.0)=16`, `(8000,12000,1.0)=4`, `(0,1000,1.0)=2`, `(0,2500,1.0)=2`, `(0,3500,1.0)=4`.

### 5.6 Summary and formatting helpers (pure, in `jev_gate.py`)
- `summarize_gate(records) -> dict` with exactly the keys `n_chunks, n_full, n_reduced, frames_sent, frames_full_equiv, frames_saved_pct, jev_time_s, vlm_time_s, reasons`. `frames_saved_pct = (equiv - sent)/equiv*100` or `None` if `equiv == 0`; `vlm_time_s` sums only non-`None` values; `reasons` maps each reason present to its count.
- `format_time_range(start_ms, end_ms) -> str`: `"{m}:{ss}-{m}:{ss}"` with whole-second floor, `ss` zero-padded to 2 digits, minutes unbounded, ASCII hyphen. `(0,5000)` → `"0:00-0:05"`; `(65000,130500)` → `"1:05-2:10"`.
- `format_tick_line(tick) -> str`: ok → `f"t={t_ms}ms ok top={name}:{p:.2f} [" + " ".join(f"{h}={probs[h]:.2f}" for h in HAZARDS) + "]"` where `name` is the highest-probability hazard (ties → first in `HAZARDS` order); error → `f"t={t_ms}ms error error={error}"`.
- `tick_to_dict(tick)` → keys `tick_id, t_ms, window_start_ms, num_frames, status, probs, error, jev_time_s`. `gate_record_to_dict(rec, include_ticks)` → keys listed in §6.4 for a `chunk` line **without** `type`, plus `"ticks": [tick_to_dict(...)]` when `include_ticks`. Neither includes thumbnails; both are `json.dumps`-safe.

---

## 6. Data model

### 6.1 `TickRecord` (add to `app/core/types.py`, mutable dataclass)
`tick_id: int`, `t_ms: int`, `window_start_ms: int`, `num_frames: int`, `status: str` (`"ok"` | `"error"`), `probs: dict[str, float] | None = None`, `error: str | None = None`, `jev_time_s: float = 0.0`, `thumbs: list[bytes] = field(default_factory=list)`.

### 6.2 `GateRecord` (add to `app/core/types.py`, mutable dataclass; one per chunk)
| Field | Type | Meaning |
| --- | --- | --- |
| `chunk_id` | int | Same as `VLMEvent.chunk_id` |
| `start_ms`, `end_ms` | int | Chunk window |
| `tier` | str | `"FULL"` / `"REDUCED"` |
| `reason` | str | One of the six reasons (§5.4) |
| `vec` | dict[str,float] \| None | Per-hazard max (None if incomplete) |
| `pmax` | float \| None | `max(vec.values())` |
| `rising_hazard` | str \| None | Only for `rising` |
| `tick_ids` | list[int] | Assigned tick ids |
| `ticks` | list[TickRecord] | Assigned ticks; `thumbs` kept **only** on the top tick, `[]` on the others |
| `top_tick_id` | int \| None | Ok tick with the highest `pmax` (ties → earliest); `None` if none are ok |
| `num_ticks_ok`, `num_ticks_error` | int | Counts |
| `sample_fps` | float | Tier fps actually used |
| `frames_sent` | int | `len(chunk.frames)` actually sent |
| `frames_full_equiv` | int | `target_frame_count(start_ms, end_ms, vlm_fps)` — what baseline plans to send |
| `jev_time_s` | float | Sum of the chunk's ticks' `jev_time_s` |
| `vlm_time_s` | float \| None | `event.latency_s` |
| `vlm_status` | str \| None | `event.status` |
| `vlm_score` | float \| None | `event.score` |
| `wall_done_s` | float | `perf_counter()` since the run's loop start, taken right after `process_chunk` returns. Non-decreasing. Includes JEV time. |
| `hold_remaining_after` | int | From the decision |

The worker MUST NOT modify a record after emitting it.

### 6.3 `SessionModel` additions (`app/models/session_model.py`)
- Attributes in `__init__`: `self.gate_records: list[GateRecord] = []`, `self.gate_meta: dict = {}`.
- `add_gate_record(rec)`: replace the record with the same `chunk_id` in place, otherwise append.
- `get_gate_by_chunk_id(chunk_id) -> GateRecord | None`.
- `clear()`: additionally `self.gate_records.clear()` and `self.gate_meta = {}`.
- Import `GateRecord` from `app.core.types`. Nothing else changes.

### 6.4 JSONL run log (`GateLog`, §7.3)
One JSON object per line, flushed after every write, UTF-8, `json.dumps(obj, separators=(",", ":"), allow_nan=False)`. Key order is exactly as listed. Path: `make_log_path(log_dir, media_path, now)` with `log_dir = os.path.join(config.cache_dir, "gate_logs")`.

| `type` | Keys (after `"type"`) | Written |
| --- | --- | --- |
| `run_start` | `time` (ISO-8601 local), `media`, `duration_ms`, `chunk_s`, `overlap_s`, `vlm_fps`, `n_chunks`, `n_ticks`, `gate_config` (`GateConfig.to_dict()`), `decider` (label string) | once, before the first tick |
| `tick` | `chunk_id`, then `tick_to_dict` keys | right after each tick |
| `chunk` | `chunk_id, start_ms, end_ms, tier, reason, vec, pmax, rising_hazard, tick_ids, num_ticks_ok, num_ticks_error, sample_fps, frames_sent, frames_full_equiv, jev_time_s, vlm_time_s, vlm_status, vlm_score, wall_done_s, hold_remaining_after, top_tick_id` | after the VLM event returned |
| `run_end` | `outcome` (`finished`/`canceled`/`error`), `chunks_done`, `wall_s`, `error` | once, last |

`gate_record_to_dict(rec, False)` yields the `chunk` keys in this order. A log write failure never aborts processing (§10).

### 6.5 JSON export
`MainWindow._gate_export_meta()` returns `None` when `session.gate_records` is empty (export identical to baseline); otherwise `{"gate": {"config": ..., "run": session.gate_meta, "summary": summarize_gate(records), "records": [gate_record_to_dict(r, True) ...]}}`, passed as `extra_meta` (F-10). CSV export is unchanged (D-10).

---

## 7. Components

### 7.1 `GatedBatchWorker.run()` — exact algorithm
Subclass of `BatchWorker` (inherits `event_ready`, `progress`, `finished`, `canceled`, `error_occurred`, `cancel()`, `_is_canceled`, `media_path`, `adapter`, `chunk_s`, `overlap_s`, `vlm_fps`). Constructor: `__init__(self, media_path, adapter, chunk_s=5.0, overlap_s=1.0, vlm_fps=4.0, parent=None, *, decider, gate_config, log_dir)` — the three keyword-only parameters are required; the worker is only ever constructed with a decider (§8.1). Adds signals `gate_ready = Signal(object)` (a `GateRecord`), `gate_run_info = Signal(object)` (a `dict`), `gate_status = Signal(str)`. `_is_hermes = hasattr(self.adapter.vlm, "start_session")`.

1. **Image input:** if `cv2.imread(self.media_path) is not None`: `return super().run()` (baseline behaviour; no JEV, no gate signals, no log).
2. Open `cap = cv2.VideoCapture(media_path)`. If not opened: `error_occurred.emit(f"Failed to open video: {media_path}")`; return. (No session started.)
3. `try: validate_against_vlm_fps(gate_config, vlm_fps)` — on `ValueError`: release `cap`, `error_occurred.emit(str(e))`, return.
4. `source_fps = cap.get(CAP_PROP_FPS) or 30.0`; `duration_ms = int(round(cap.get(CAP_PROP_FRAME_COUNT)/source_fps*1000.0))`; `windows = compute_chunk_windows(duration_ms=..., chunk_s=..., overlap_s=..., include_partial=True)`. If empty: release, `finished.emit()`, return.
5. `tick_times = compute_tick_times(duration_ms, cfg.window_ms, cfg.stride_ms)`; `assigned = assign_ticks_to_chunks(tick_times, windows)`.
6. Open `GateLog(make_log_path(log_dir, media_path, datetime.now()))`; write `run_start`; emit `gate_run_info` with `{"media","duration_ms","chunk_s","overlap_s","vlm_fps","n_chunks","n_ticks","gate_config","decider","log_path"}`. `decider` label = `getattr(decider, "label", type(decider).__name__)`.
7. `if _is_hermes: adapter.vlm.start_session()` — everything after this line is inside `try/except Exception`.
8. `router = GateRouter(cfg)`; `start_perf = perf_counter()`; for `idx, (w_start, w_end)` in `windows`:
   a. If `_is_canceled`: break.
   b. For each `tick_id` in `assigned[idx]`: if `_is_canceled`: break the loop; emit `gate_status(f"JEV tick {tick_id+1}/{n_ticks} · chunk {idx+1}/{total}")`; `rec = evaluate_tick(...)`; write the `tick` line; collect.
   c. If `_is_canceled`: break (the chunk is **not** sent to the VLM).
   d. `decision = router.decide(collected)`.
   e. `fps = vlm_fps if decision.tier == "FULL" else cfg.route_reduced_fps`; sample (§5.5); build `VideoChunk(chunk_id=idx, source=media_path, start_ms=w_start, end_ms=w_end, fps=fps, frames=..., timestamps_ms=...)`.
   f. `gate_status.emit(f"VLM chunk {idx+1}/{total} · {tier} @ {fps:g} fps ({reason})")`; `event = adapter.process_chunk(chunk, event_id=idx+1)`.
   g. Build the `GateRecord` (§6.2), keep thumbnails on the top tick only; write the `chunk` line; **emit `gate_ready`, then `event_ready`**; `completed = idx+1`; ETA exactly as baseline (`elapsed/completed*remaining`); `progress.emit(completed, total, eta_s)`.
9. After the loop: `cap.release()`; `if _is_hermes: end_session()`; write `run_end` (`canceled` if `_is_canceled` else `finished`) and close the log; emit `canceled` or `finished`.
10. On any `Exception` in steps 7–9: `end_session()` guarded by `try/except`, `cap.release()`, write `run_end` with `outcome="error"` and `error=str(e)`, close the log, `error_occurred.emit(str(e))`.

### 7.2 Change budget (exhaustive; anything else is forbidden)
**New files**

| File | Content |
| --- | --- |
| `app/core/jev_gate.py` | §5.1, §5.3, §5.4, §5.6, `GateConfig`, constants |
| `app/core/frame_sampler.py` | §5.5 functions + `resize_max_side`, `encode_jpeg`, `make_thumb` |
| `app/core/jev_adapter.py` | `JevDecider`, `validate_probs`, `evaluate_tick`, `OneJevDecider` (in-process, Appendix A), `HAZARD_QUESTIONS` (verbatim from `hazard_monitor.HAZARDS`) |
| `app/core/gate_log.py` | `GateLog`, `make_log_path` |
| `app/workers/gated_batch_worker.py` | §7.1 |
| `app/widgets/gate_panel.py` | §9 |
| `examples/mock_jev.py` | `MockJevDecider`, `DEMO_RULES` |
| `tests/support.py` | `make_test_video`, `RecordingAdapter` |
| `tests/test_gate_core.py`, `test_frame_sampler.py`, `test_jev_adapter.py`, `test_gate_log.py`, `test_gated_worker.py`, `test_gate_ui.py`, `test_gate_main_window.py`, `test_compare_latency_3way.py` | §12 |
| `scripts/g1_report.py` | Appendix B analysis over a GateLog JSONL (CPU-only, no app imports; delivered in Phase 5, run by the developer) |

**Edited files (exact edits only)**

| File | Edit |
| --- | --- |
| `app/core/types.py` | Append `TickRecord`, `GateRecord` (§6.1–6.2). |
| `app/models/session_model.py` | §6.3. |
| `app/main_window.py` | §8.1. |
| `main.py` | §8.2. |
| `compare_latency.py` | §11 (last phase only). |

No other file is touched (I-3). In particular `app/config.py` is **not** changed: `GateConfig` is self-contained (D-8).

### 7.3 Interfaces (exact signatures)
```python
# app/core/jev_gate.py
HAZARDS: tuple[str, ...] = ("pedestrian_vehicle", "no_ppe", "blocked_exit", "unsafe_load", "fall_or_spill")
TIER_FULL, TIER_REDUCED = "FULL", "REDUCED"
REASON_INCOMPLETE, REASON_ABOVE_FULL, REASON_ABOVE_WATCH = "jev_incomplete", "above_full", "above_watch"
REASON_RISING, REASON_HOLD, REASON_CLEAR = "rising", "hold", "clear"
EPS = 1e-9

@dataclass(frozen=True)
class GateConfig:
    jev_window_s: float = 3.0
    jev_stride_s: float = 1.5
    jev_sample_fps: float = 4.0
    jev_max_side: int = 640
    route_full_threshold: float = 0.35
    route_watch_threshold: float = 0.20
    route_rise_delta: float = 0.10
    route_hold_chunks: int = 2
    route_reduced_fps: float = 1.0
    # __post_init__ raises ValueError unless ALL hold:
    #   jev_window_s > 0, jev_stride_s > 0, jev_stride_s <= jev_window_s, jev_sample_fps > 0,
    #   jev_max_side >= 32 (int), 0 < route_watch_threshold <= route_full_threshold <= 1,
    #   0 < route_rise_delta <= 1, route_hold_chunks >= 0 (int, not bool), route_reduced_fps > 0,
    #   window_ms >= 1 and stride_ms >= 1.
    @property
    def window_ms(self) -> int: ...      # int(round(jev_window_s * 1000))
    @property
    def stride_ms(self) -> int: ...      # int(round(jev_stride_s * 1000))
    def to_dict(self) -> dict: ...       # dataclasses.asdict

def validate_against_vlm_fps(cfg: GateConfig, vlm_fps: float) -> None
    # ValueError("route_reduced_fps (X) must be lower than vlm_fps (Y)") if cfg.route_reduced_fps >= vlm_fps

def compute_tick_times(duration_ms: int, window_ms: int, stride_ms: int) -> list[int]
def assign_ticks_to_chunks(tick_times: list[int], windows: list[tuple[int, int]]) -> list[list[int]]

@dataclass(frozen=True)
class RouteDecision:
    tier: str; reason: str; vec: dict[str, float] | None; pmax: float | None
    rising_hazard: str | None; hold_remaining_after: int

class GateRouter:
    def __init__(self, cfg: GateConfig): ...
    def decide(self, ticks: Sequence[TickRecord]) -> RouteDecision: ...
    hold_remaining: int          # read-only for callers
    prev_vec: dict[str, float] | None

def summarize_gate(records: Sequence[GateRecord]) -> dict
def format_time_range(start_ms: int, end_ms: int) -> str
def format_tick_line(tick: TickRecord) -> str
def tick_to_dict(tick: TickRecord) -> dict
def gate_record_to_dict(rec: GateRecord, include_ticks: bool) -> dict

# app/core/frame_sampler.py   (numpy + cv2 only)
def target_frame_count(start_ms: int, end_ms: int, fps: float) -> int
def sample_timestamps(start_ms: int, end_ms: int, fps: float) -> list[int]
def read_frames_at(cap, timestamps_ms: list[int], rgb: bool = True) -> tuple[list[np.ndarray], list[int]]
def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray
    # if max(h,w) <= max_side: return img unchanged; else scale = max_side/max(h,w),
    # new_w = max(1,int(w*scale)), new_h = max(1,int(h*scale)), cv2.resize(..., INTER_AREA)
def encode_jpeg(img_bgr: np.ndarray, quality: int = 85) -> bytes      # cv2.imencode(".jpg", ...); raise RuntimeError if it fails
def make_thumb(img_bgr: np.ndarray, max_side: int = 160, quality: int = 70) -> bytes
    # resize_max_side + encode_jpeg

# app/core/jev_adapter.py
class JevDecider(Protocol):
    def __call__(self, frames_rgb: list[np.ndarray], fps: float, t_ms: int) -> dict[str, float]: ...
    # t_ms is context for logging/tests only; a real decider MUST NOT use it to decide.

def validate_probs(raw: object) -> dict[str, float]
    # ValueError("invalid_output: <reason>") unless raw is a dict whose key set == set(HAZARDS)
    # and every value is int|float (not bool), finite, 0.0 <= v <= 1.0. Returns {h: float(raw[h]) for h in HAZARDS}.

def evaluate_tick(decider, cap, tick_id: int, t_ms: int, cfg: GateConfig) -> TickRecord   # §5.2

HAZARD_QUESTIONS: dict[str, str]
    # The five statement texts, COPIED VERBATIM from HAZARDS in the supplied hazard_monitor.py
    # ("A person is in danger of being struck by a forklift or other moving vehicle", ...).

class OneJevDecider:
    label = "onejev"
    def __init__(self, model_path: str): ...
        # In-process OneJev-0.8B (C-11, D-16, D-18; normative algorithm in Appendix A).
        # 1. import torch/transformers HERE (lazy, I-5).
        # 2. If not torch.cuda.is_available(): raise RuntimeError(
        #        "OneJev requires CUDA (--jev onejev); no CUDA device is available").
        # 3. If not os.path.isdir(model_path): raise RuntimeError(f"OneJev model not found: {model_path}").
        # 4. processor = AutoProcessor.from_pretrained(model_path)
        #    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        #        model_path, dtype=torch.bfloat16).to("cuda"); model.eval()
        # 5. slot check: tokenizer must encode "A" and "B" to exactly one token id each
        #    (id_a, id_b); otherwise RuntimeError("slot letters are not single tokens").
    def __call__(self, frames_rgb, fps, t_ms) -> dict[str, float]: ...
        # Appendix A: one forward pass per hazard question; p_yes = softmax over the two
        # slot-letter logits (T = 1.0); returns {hazard: float}. validate_probs is applied
        # by evaluate_tick, not here.

# app/core/gate_log.py
def make_log_path(log_dir: str, media_path: str, now: datetime) -> str
    # basename = os.path.basename(media_path.replace("\\", "/")); stem = os.path.splitext(basename)[0];
    # stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem) or "media"
    # return os.path.join(log_dir, f"gate_{now:%Y%m%d_%H%M%S}_{stem}.jsonl")
class GateLog:
    def __init__(self, path: str): ...     # makes parent dirs; opens for write; never raises OSError (sets failed/error)
    def write(self, obj: dict) -> bool: ... # json.dumps errors propagate; OSError -> failed=True, returns False
    def close(self) -> None: ...
    path: str; failed: bool; error: str | None

# examples/mock_jev.py
DEMO_RULES = [(5500, 8000, {"no_ppe": 0.25}), (10000, 12000, {"no_ppe": 0.60})]
class MockJevDecider:
    label = "mock"
    def __init__(self, rules=None, default: float = 0.05, delay_s: float = 0.0,
                 fail_at=(), bad_output_at=()): ...
        # rules=None -> DEMO_RULES; rules=[] -> no rules. Rule = (start_ms, end_ms, {hazard: p});
        # first rule with start_ms <= t_ms <= end_ms wins; unlisted hazards get `default`.
        # t_ms in fail_at -> raise RuntimeError(f"mock failure at t={t_ms}")
        # t_ms in bad_output_at -> return the dict without the "blocked_exit" key
        # sleeps delay_s first (time.sleep) when delay_s > 0
    def __call__(self, frames_rgb, fps, t_ms) -> dict[str, float]: ...
        # Deterministic: output depends only on t_ms (rules + default);
        # frames_rgb is accepted but not inspected (D-17).
    @classmethod
    def from_json(cls, path: str) -> "MockJevDecider": ...
        # JSON object: {"default": float, "delay_s": float, "rules": [[start,end,{...}],...],
        #               "fail_at": [ints], "bad_output_at": [ints]}; all keys optional;
        # unknown keys -> ValueError.
```
`jev_gate.py` imports only `dataclasses`, `typing`, `math` and `app.core.types`. It MUST NOT import cv2, torch or PySide6.

---

## 8. Wiring

### 8.1 `app/main_window.py` — the only permitted edits
1. Imports (top): `GatedBatchWorker`, `GatePanel`, `GateConfig`, `summarize_gate`, `gate_record_to_dict`, `GateRecord`.
2. `__init__`: append parameters **after** `parent=None`: `jev=None, gate_config=None`. Store `self.jev = jev`, `self.gate_config = gate_config if gate_config is not None else GateConfig()`, `self.gate_panel = None` (before `_init_ui()` is called).
3. `_init_ui`: after `right_layout.addWidget(self.event_feed, stretch=1)` add: `if self.jev is not None: self.gate_panel = GatePanel(self.config, self); right_layout.addWidget(self.gate_panel, stretch=0)`; connect `self.gate_panel.chunk_selected` to `_on_gate_chunk_selected` in `_connect_signals` (guarded by `if self.gate_panel is not None`).
4. `_start_processing`: after `self.event_feed.clear()` add `if self.gate_panel is not None: self.gate_panel.clear()`. Replace the seven-line `self.batch_worker = BatchWorker(...)` statement with `self.batch_worker = self._make_batch_worker(chunk_s, overlap_s, vlm_fps)`. After the five existing `connect` lines and before `start()` add `self._connect_gate_signals()`.
5. New methods:
   - `_make_batch_worker(chunk_s, overlap_s, vlm_fps)`: if `self.jev is None` return exactly `BatchWorker(media_path=self.session.media_path, adapter=self.adapter, chunk_s=chunk_s, overlap_s=overlap_s, vlm_fps=vlm_fps, parent=self)`; else `GatedBatchWorker(media_path=..., adapter=..., chunk_s=..., overlap_s=..., vlm_fps=..., parent=self, decider=self.jev, gate_config=self.gate_config, log_dir=os.path.join(self.config.cache_dir, "gate_logs"))`.
   - `_connect_gate_signals()`: if `isinstance(self.batch_worker, GatedBatchWorker)`: connect `gate_ready` → `_on_gate_record_ready`, `gate_run_info` → `_on_gate_run_info`, `gate_status` → `self.status_bar.showMessage`.
   - `_on_gate_record_ready(rec)`: `self.session.add_gate_record(rec)`; `self.gate_panel.add_record(rec)`.
   - `_on_gate_run_info(info)`: `self.session.gate_meta = info`.
   - `_on_gate_chunk_selected(chunk_id)`: if `self.current_state == "READY"`, `ev = self.session.get_event_by_chunk_id(chunk_id)`; if `ev` is not None: `self._on_seek_requested(ev.start_ms)`. Otherwise do nothing.
   - `_gate_export_meta()`: §6.5.
6. `_export_json`: add `extra_meta=self._gate_export_meta()` to the existing `export_json` call. Nothing else.
`closeEvent`, all other methods and all existing signal connections stay untouched.

### 8.2 `main.py` — the only permitted edits
- New arguments: `--jev {off,mock,onejev}` (default `off`); `--jev-model-path PATH` (default: the OneJev-0.8B snapshot under `F:\URCA_PROJECTS\.hf_cache\hub` — the exact snapshot path is recorded in the Phase 0 environment record and pinned as this default, D-19); `--mock-jev-script PATH`; `--route-full-thr FLOAT`; `--route-watch-thr FLOAT`; `--route-rise-delta FLOAT`; `--route-hold-chunks INT`; `--route-reduced-fps FLOAT` (all route flags default `None`).
- After `parse_args()` and **before** `QApplication(...)`:
  - `--jev off` with any `--route-*` or `--mock-jev-script` provided → `parser.error("... require --jev mock|onejev")`.
  - `--mock-jev-script` with `--jev onejev` → `parser.error`.
  - `--jev-model-path` without `--jev onejev` → `parser.error("--jev-model-path requires --jev onejev")`.
  - Build `GateConfig(**only the provided overrides)`; `ValueError` → `parser.error(str(e))`.
  - `mock` → `MockJevDecider.from_json(path)` if a script is given, else `MockJevDecider()`; `ValueError`/`OSError` → `parser.error(str(e))`.
  - `onejev` → `OneJevDecider(model_path=args.jev_model_path)`; `RuntimeError` (no CUDA, missing model, bad slot letters) → `parser.error(str(e))` (exit code 2). The model loads onto the GPU here, before the window opens — the same pattern as the HERMES runner.
- Append ` + JEV(mock)` / ` + JEV(onejev)` to `model_label` when a decider is active.
- Call `MainWindow(vlm=runner, model_label=model_label, config=DEFAULT_CONFIG, jev=decider, gate_config=gate_cfg)` (`None`/`None` when off).

---

## 9. UI: `GatePanel` (C-2)
Shown only when a decider is configured. Layout, top to bottom, inside one `QWidget` (`setMinimumHeight(260)`):
1. **Summary label:** `f"FULL {n_full} | REDUCED {n_reduced} | frames {sent}/{equiv} ({pct:.1f}% fewer) | JEV {jev:.1f}s | VLM {vlm:.1f}s"`; when `equiv == 0`: `"frames 0/0 (n/a)"` in place of the frames segment; with no records: `"No gate data yet"`.
2. **Table** (`QTableWidget`, read-only, single row selection, sorting disabled). Headers exactly: `#`, `Time`, `Tier`, `Reason`, `pmax`, `Top hazard`, `Frames`, `JEV s`, `VLM s`, `VLM score`. Cells: `str(chunk_id+1)`; `format_time_range`; tier; reason; `pmax` as `f"{x:.2f}"` or `"-"`; top hazard from `vec` (ties → `HAZARDS` order) or `"-"`; `f"{sent}/{equiv}"`; `f"{jev_time_s:.1f}"`; `f"{vlm_time_s:.1f}"` or `"-"`; `f"{vlm_score:.2f}"` or `"-"`.
   The **Tier** cell background by reason: `above_full` → `config.color_hazard`; `above_watch`, `rising` → `config.color_warning`; `hold` → `config.color_primary`; `jev_incomplete` → `config.color_off`; `clear` → `config.color_safe`. The reason string is also stored in `Qt.UserRole` of that cell. Colour is never the only signal (tier and reason are text).
   A record for an existing `chunk_id` updates its row in place; otherwise a row is appended.
3. **Detail area** for the selected row: a text label with one `format_tick_line` per tick (or `"no JEV ticks"`), and up to three thumbnail labels (`top tick` thumbs, scaled to height 72). Without thumbnails: `"no JEV thumbnails"`.

API: `chunk_selected = Signal(int)` (emitted with the chunk id when the selected row changes), `clear()`, `add_record(rec)`, `summary_text() -> str`, `detail_text() -> str`, `thumb_count_shown() -> int`, attribute `table`.
VLM-side frames remain visible in the existing `EventFeed` thumbnails, so both what JEV saw and what HERMES saw can be compared. No overlay is added to `VideoView` or the timeline (D-9).

---

## 10. Failure handling

### 10.1 No timeouts (C-5, precise meaning)
- No code path may abandon, time-limit, or race a JEV or VLM call. A hung call hangs the worker; the user cancels with the existing Cancel button (takes effect at the next cancel check, §7.1).
- The status bar always shows what the worker is doing (`gate_status` is emitted **before** each tick and each VLM call), so a stall is visible.
- The words `timeout`, `QTimer`, `wait_for`, `alarm(`, `deadline`, `setTimeout` (case-insensitive) MUST NOT appear in any **new** source file, not even in comments (test T-1). The existing `wait(1000)` in `MainWindow.closeEvent` is pre-existing and untouched.

### 10.2 Behaviour table
| Situation | Behaviour |
| --- | --- |
| Decider raises `Exception` | Tick `error` (`"<Type>: <msg>"`), chunk `FULL/jev_incomplete`, run continues. |
| Decider returns invalid output | Tick `error` (`invalid_output: ...`), same as above. |
| Tick has no readable frames | Tick `error` (`no_frames`), same as above. |
| Chunk has no assigned ticks | `FULL/jev_incomplete`. |
| `KeyboardInterrupt`, `SystemExit` in a decider | Propagate (not caught). |
| VLM returns `error`/`invalid` event | Recorded in `GateRecord.vlm_status`; run continues (adapter behaviour unchanged). |
| Gate log write fails | `GateLog.failed=True`, further writes skipped, one `gate_status` message `"gate log write failed: <error>"` (emitted once); processing continues. |
| Video cannot be opened | `error_occurred("Failed to open video: <path>")`, no session started. |
| `route_reduced_fps >= vlm_fps` | `error_occurred("route_reduced_fps (X) must be lower than vlm_fps (Y)")`, nothing started. |
| OneJev CUDA unavailable / model missing / slot letters not single tokens | `RuntimeError` from `OneJevDecider.__init__` → `parser.error` at startup (§8.2); nothing starts. |
| Unexpected `Exception` in the loop | §7.1 step 10. |
| Cancel during ticks | Chunk not sent; `canceled` emitted; already emitted chunks preserved. |
| Image input | Baseline behaviour (§7.1 step 1). |

---

## 11. Final step: `compare_latency.py` — three systems

### 11.1 Scope (D-20, D-21)
The three engines are **HERMES**, **InfiniPot-V** (both unchanged), and **JEV+HERMES** (the gated system of this document), per C-8. The change happens in Phase 6 only, after every earlier phase is verified. Existing engine names, worker code, output files and behaviour stay untouched; `temper/` stays because the InfiniPot worker imports from it (F-13).

### 11.2 The `jev_hermes` worker
Add `run_jev_hermes_worker` and extend `--engine` choices with `jev_hermes`. It mirrors `run_hermes_worker` exactly (same clean-subprocess isolation, same eviction of `temper/` from `sys.path`/`sys.modules`, same `sample_video_chunks` output, same model-loading arguments) plus:
1. Loads `OneJevDecider(args.jev_model_path)` **in-process** in the same subprocess (C-11). CUDA is required: if `torch.cuda.is_available()` is false, or the model path is missing, the worker **fails loudly** (non-zero exit with the message). It never substitutes a mock (D-21) — a mock latency would be a false benchmark.
2. Computes `tick_times = compute_tick_times(...)` and `assigned = assign_ticks_to_chunks(...)` once, exactly as §7.1.
3. Per chunk, **sequentially**:
   - For each assigned tick: `evaluate_tick(decider, cap, ...)` against its own `cv2.VideoCapture` handle (the same evaluate_tick the app uses), accumulating `jev_time_s`. Tick frames are read fresh from the file at `jev_sample_fps` and resized to `jev_max_side` — the pre-resized 448 chunk frames are **not** reused, because the app's JEV sees ≤640-px frames, and the benchmark must measure what the app does.
   - `router.decide(...)` → tier/reason (a single `GateRouter` created before the chunk loop, its state carried across chunks exactly as in the app).
   - `FULL` → use the chunk's `sample_video_chunks` frames (bit-identical to the `hermes` engine's input). `REDUCED` → re-sample the window at `route_reduced_fps` from the video and resize each frame with the **same** `--max-dim` resize `sample_video_chunks` uses, so HERMES input differs from the `hermes` engine only in frame count.
   - HERMES stages run and are measured with the **identical instrumentation** as `run_hermes_worker` (same keys, same timers).
4. Per-chunk record: **all existing keys** (`total_time_s, ingest_time_s, compress_time_s, gen_time_s, latency_per_frame_ms, kv_cache_len, peak_vram_mb, caption, score`) plus `jev_time_s`, `jev_tier`, `jev_reason`, `jev_pmax`, `frames_sent`, `frames_full_equiv`, `net_total_time_s`.

### 11.3 Accounting (D-20)
- `total_time_s` is **gross**: JEV time + HERMES stages, measured sequentially (which is also how the app runs them, §4).
- `net_total_time_s = total_time_s − jev_time_s` is recorded alongside, so the pure HERMES cost at the routed frame rate is visible.
- The report table gains a third column and a `JEV s` row; the summary line prints, for the gated engine, `frames_sent/frames_full_equiv` and the savings percentage. Exact formatting is fixed by the dry-run test (§12.1 3WAY-1).
- Output file: `jev_hermes_benchmark_telemetry.json`. The existing two output names are unchanged. Benchmarks are run **from a scratch directory** (F-14/defect #5).
- `--jev-model-path` is added to the master's worker-spawn arguments (`--engine jev_hermes` only).

### 11.4 What the agent delivers
The code, a dry-run test with **injected fake runner/decider classes** (no GPU, no model) proving the JSON plumbing, the table rendering and `net = gross − jev`, and the exact commands for the developer's GPU run (DEV-4). The real three-way numbers are **NOT VERIFIED** by the agent.

---

## 12. Verification

### 12.1 Test catalogue (unittest style, under `vlm_test_ui/tests/`, matching the repo)
Every new test: no network, no GPU, no absolute paths (media relative to the test file), pure logic tested without a `QApplication`.

| ID | Subject |
| --- | --- |
| R-1 | Existing headless tests still pass (11 as of baseline). |
| R-2 | Mock batch flow on `synthetic_test.mp4` still gives 3 chunks / 20-20-15 frames. |
| R-3 | With all new flags off, `MainWindow` behaves as before (EMPTY→LOADED→PROCESSING→READY). |
| TICK-1..5 | `compute_tick_times`: the four examples in §5.1; `ValueError` on `window_ms<1`, `stride_ms<1`, `stride_ms>window_ms`; `duration_ms<=0` → `[]`; forced append of `duration_ms`; strictly ascending output. |
| ASSIGN-1..4 | Region boundary (`t == e_k` → chunk k); tick beyond last window → last chunk; chunk with empty list; empty windows → `[]`. |
| ROUTE-1..9 | Each router rule 1–6 fires; precedence by rule order; EPS edge (`pmax == thr − 1e-10` does not fire); `hold` countdown across chunks and reset on next `above_full`; `prev_vec` becomes `None` after `jev_incomplete` and is copied otherwise; `rising_hazard` = first in `HAZARDS` order; router inputs not mutated; `hold_remaining_after` correct. |
| SAMPLER-1..7 | `target_frame_count`: the seven expected pairs in §5.5; `sample_timestamps` endpoints inclusive; `read_frames_at` stops at first failed read, single fallback read, empty-timestamps → `([], [])`, `rgb` flag channel order; `resize_max_side` no-op ≤ max side and ≥1 px outputs; `encode_jpeg` failure → `RuntimeError`; `make_thumb` count ≤ 3. |
| EVAL-1..6 | `evaluate_tick`: success path fields; `no_frames` → decider **not called** (recording decider proves it); decider exception → `"<Type>: <msg>"`; invalid output → `invalid_output:` prefix; `jev_time_s` ≥ decider's `delay_s`; `KeyboardInterrupt` propagates. |
| VALID-1..5 | `validate_probs`: exact key set; `bool` rejected; NaN/inf rejected; out-of-range rejected; ints accepted as floats. |
| MOCK-1..4 | `MockJevDecider`: DEMO_RULES windows and defaults; first-rule-wins; `fail_at` raises; `bad_output_at` drops `blocked_exit`; `from_json` round-trip and unknown-key `ValueError`. |
| JEV-8 | **Lazy-import audit (I-5):** after an R-3 run (`--jev off`) and a MOCK run, `torch` and `transformers` are absent from `sys.modules`. |
| LOG-1..4 | `make_log_path` sanitisation; `GateLog` flush-per-write; `OSError` → `failed=True`, returns `False`, never raises; `gate_record_to_dict` key order matches §6.4. |
| GATE-1..6 | `GatedBatchWorker` on a synthetic video (made by `tests.support.make_test_video`, so every planned timestamp decodes) with `MockJevDecider`: **GATE-1** chunk count == `compute_chunk_windows` (I-6); **GATE-2** each chunk emitted exactly once, in order, `event_id = idx+1`; **GATE-3** REDUCED chunks carry `target_frame_count(s,e,reduced_fps)` frames and FULL chunks exactly the baseline counts; **GATE-4** image input → `super().run()` (no gate signals, no log file); **GATE-5** cancel during ticks → chunk not sent, `canceled` emitted, prior records preserved; **GATE-6** log contains `run_start`, one `tick` per tick, one `chunk` per chunk, `run_end` last, with the §6.4 key orders. |
| WIN-1 | `jev=None` → `_make_batch_worker` returns a plain `BatchWorker`; full-run equivalence with baseline events (with R-2). |
| UI-1..3 | `GatePanel`: summary text formats (no-records, `equiv==0`, normal); row update-in-place on same `chunk_id`; detail text and thumbnail count; `chunk_selected` emitted on row change. Offscreen (`QT_QPA_PLATFORM=offscreen`). |
| MAINW-1..3 | `--jev off` → no `gate_panel`, plain worker; gate signals connected only for `GatedBatchWorker`; `_gate_export_meta()` `None` when empty, correct dict shape when not, merged into the export (F-10). |
| MAIN-1..2 | `main.py` validation errors (`route-*`/script flags with `--jev off`; `--jev-model-path` without `--jev onejev`; script with `onejev`); `model_label` suffix. |
| ONEJEV-1 (**DEV**) | Contract test, `skipUnless` model snapshot exists **and** CUDA available: `OneJevDecider` loads, one synthetic frame set → all five probs in [0,1]; same input twice → identical probs; red frame vs blue frame produce different probs. Run by the developer. |
| T-1 | **No-timeout scan** (§10.1 word list) over all new files and the `git diff` additions. |
| 3WAY-1..2 | `jev_hermes` benchmark plumbing with injected fake runner/decider: all telemetry keys present, `net_total_time_s == total_time_s − jev_time_s`, 3-column table renders, existing engine outputs untouched. |

### 12.2 Fail-on-break proof
Each new test is shown once to fail when the behaviour it guards is broken (e.g. drop the BGR→RGB conversion → SAMPLER/EVAL colour check fails; invert the rule-4 delta sign → ROUTE-rising fails; skip `evaluate_tick`'s decider guard → EVAL-2 fails). Paste both outputs in the phase report.

### 12.3 Evidence rule
A phase is complete only when the report contains the exact command, its pasted output, the baseline comparison (R-1..R-3 re-run), the `git diff --stat`, the T-1 scan output, and an explicit **NOT VERIFIED** list. "Should work" is not evidence.

---

## 13. Implementation phases (stop after every phase — D-15)

| Phase | Content | Files | Gate |
| --- | --- | --- | --- |
| 0 | Environment record (venv `F:\URCA_PROJECTS\.venv`: Python 3.13.5, transformers 5.17.0, `Qwen3_5ForConditionalGeneration` import, torch 2.14.0+cu126, CUDA device via `nvidia-smi`); download OneJev-0.8B into `.hf_cache` (agent action, pre-approved — D-19; a Hub-side transfer error aborted an earlier attempt — `snapshot_download` resumes) and pin the snapshot path as the `--jev-model-path` default; baseline R-1..R-3 captured at `fedfb07`; register confirmed (done 5 Oct 2026). **No code.** | none | Baseline report |
| 1 | `jev_gate.py` + `frame_sampler.py`; append `TickRecord`/`GateRecord` to `types.py` (§7.3's `jev_gate` imports them); `tests/support.py` (`make_test_video`, `RecordingAdapter`); tests. | 3 new + 1 edited + 2 test files | TICK, ASSIGN, ROUTE, SAMPLER |
| 2 | `jev_adapter.py`, `gate_log.py`, `mock_jev.py`; `session_model.py` additions (§6.3); tests. | 3 new + 1 edited + 2 test files | EVAL, VALID, MOCK, JEV-8, LOG |
| 3 | `gated_batch_worker.py`; tests. | 1 new + 1 test file | GATE, WIN-1 |
| 4 | `gate_panel.py`; `main_window.py` (§8.1), `main.py` (§8.2) edits; tests; offscreen screenshots. | 1 new + 2 edited + 2 test files | UI, MAINW, MAIN, T-1 |
| 5 | **DEV (GPU, developer machine):** real HERMES + real OneJev on a fully-decoding test video via the UI (`--hermes --jev onejev`); VRAM peak recorded (HERMES 4-bit + OneJev bf16 must coexist); **G0:** per-chunk `vlm_time_s` of the gated run compared against a `--jev off` baseline run of the same video (REDUCED chunks cost less; FULL chunks ≈ baseline); **G1** via `scripts/g1_report.py` on the GateLog. Agent prepares commands and the script; developer runs. | `scripts/g1_report.py` (new) | DEV-1, DEV-2, G0 + G1 reports |
| 6 | `compare_latency.py` three-way (§11); dry-run tests. | 1 edited + 1 test file | 3WAY, DEV-4 commands |

**DEV gates** (developer machine; the agent prepares commands and reports results as NOT VERIFIED until the developer pastes output): **DEV-1** = the Phase 5 UI run (real HERMES + real OneJev, synthetic video); **DEV-2** = the ONEJEV-1 contract test executed by the developer; **DEV-3** = 60-minute soak with `wall_done_s` comparison against a baseline run (§1.5; developer's timing, gates no code); **DEV-4** = the three-way benchmark run (Phase 6).

---

## 14. Decision Register

Status values: **Approved** (developer statement, quoted or dated) — all rows below are Approved as of 5 Oct 2026 unless noted. The register is the authority for what may be implemented.

| ID | Decision | Status |
| --- | --- | --- |
| D-1 | Stage 2 (live simulation, concurrent threads, lag accounting) out of scope; this document is Stage 1 only. | **Approved** — doc §1.3 |
| D-2 | Source of truth is the developer-supplied project directory (`UI_TEST` @ `fedfb07`, clean tree). Git history of other copies is irrelevant. | **Approved** — developer: "i will provide you with the latest codebase for those updates. you dont have to think about git history anymore, i will change the project directory for you to work on the new one" |
| D-3 | Question panel: the five hazard names **and texts** verbatim from `hazard_monitor.py` `HAZARDS`. | **Approved** — file supplied |
| D-4 | `GateConfig` defaults exactly as §7.3. | **Approved** |
| D-5 | "No timeout" means exactly §10.1. | **Approved** |
| D-6 | Hazard names as in `hazard_monitor.py` v0.1. | **Approved** |
| D-7 | The v0.1 trigger logic (`min_consecutive`, cooldown) is not used; routing rules §5.4 replace it. | **Approved** — doc §1.4 |
| D-8 | `GateConfig` is self-contained; `app/config.py` is not changed. | **Approved** |
| D-9 | No JEV overlay on `VideoView`, no timeline lane, no new toolbar buttons; `GatePanel` only. | **Approved** |
| D-10 | CSV export unchanged; only the JSON export gains the `gate` block via `extra_meta`. | **Approved** |
| D-11 | [VERIFIED] baseline facts re-checked against the supplied tree (done, §1.6). | **Approved** |
| D-12 | Keep the 1 s chunk overlap with the stateful HERMES runner (existing behaviour, defect #4 kept). | **Approved** |
| D-13 | G1 analysis (Appendix B) runs on the **synthetic video** with `DEMO_RULES` as pseudo-ground-truth; real-footage analysis is the developer's, later, using the same script. | **Approved** — developer: "Synthetic video only" |
| D-14 | Existing defects (§2.4) are reported, not fixed. | **Approved** |
| D-15 | The agent stops for approval after **every** phase. | **Approved** — developer: "Stop after every phase" |
| D-16 | OneJev runs **in-process** via `transformers` on **OneJev-0.8B**; no HTTP server, no ports, no `qev` package. Motivation: the code will become an `ask_qwen` function inside an NVIDIA DeepStream pipeline and must be self-contained. | **Approved** — developer: "i wanted to use it within transformers library given that it is built upon Qwen, the 0.8B class is the lightweight model we will need to work with… external HTTP ports arent interable with the full system later… we will need to design the codebase to be self contained with no HTTP or porting here" |
| D-17 | The decider interface consumes **RGB frames directly** (`frames_rgb: list[np.ndarray]`); the JPEG/data-URI encoding steps of the earlier draft (§5.2 step 4, glossary, `MockJevDecider`) are amended accordingly — they existed only for the HTTP protocol. `jev_max_side` still bounds the frames sent to the decider (matches `hazard_monitor` v0.1 `max_side=640`). | **Approved** — consequence of D-16 |
| D-18 | JEV device: **CUDA only**; `OneJevDecider.__init__` raises when CUDA is unavailable (startup error, §8.2/§10.2). | **Approved** — developer: "CUDA only" |
| D-19 | Weights: `OmniJev/OneJev-0.8B` downloaded into `F:\URCA_PROJECTS\.hf_cache` by the implementing agent during **Phase 0** (the developer pre-approves this single download job; a first attempt on 5 Oct 2026 aborted with a Hub-side transfer error — `snapshot_download` resumes); `--jev-model-path` defaults to that snapshot path, pinned in the Phase 0 environment record. | **Approved** — developer: "I pre-download to .hf_cache", then "you can ignore the download step, we will ask the coding agent to download the model in hf_cache later" |
| D-20 | Benchmark accounting: `total_time_s` **gross** (includes JEV), `net_total_time_s = total − jev`, and a `JEV s` column in the table. | **Approved** — developer accepted the recommendation |
| D-21 | The benchmark uses the **real in-process JEV only** and fails loudly when CUDA/model is unavailable; `MockJevDecider` appears only in unit tests and the dry-run plumbing test. | **Approved** — developer: "we will try to use the Real JEV but within the transformers library" |
| D-22 | Python environment: `F:\URCA_PROJECTS\.venv` (Python 3.13.5, transformers 5.17.0 — `Qwen3_5ForConditionalGeneration` import verified, torch 2.14.0+cu126, CUDA available). | **Approved** — developer: "we will operate with the one that is in .venv in URCA_PROJECTS" |
| D-23 | Benchmark engines: HERMES, InfiniPot-V, JEV+HERMES (literal reading of C-8). | **Approved** — C-8's own wording names the three systems |

---

## Appendix A — OneJev in-process scoring (normative, D-16)

The normative reference is the `qev` source copied to `references/sources/qev/ (bundled in this package)` (`prompt.py`, `media.py`, `engine.py`, `mm_engine.py`, `calibrate.py`, `answers.py`). `OneJevDecider` reproduces its **scoring semantics** with the minimum machinery: no server, no forking/caching, no CUDA graphs, no batched-fork — one plain forward pass per question (the "training evaluation path", `mm_engine._full_logits`). Deviating from the exact rendering below would break calibration; do not reword the prompt.

1. **Load (once, `__init__`):** `AutoProcessor.from_pretrained(model_path)`; `Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")`; `model.eval()`. CUDA required (D-18). Slot ids: `id_a = processor.tokenizer.encode("A", add_special_tokens=False)`, same for `"B"`; each MUST be exactly one token id, else `RuntimeError`.
2. **Media (per call, from `qev/media.py`):** each RGB np frame → `PIL.Image.fromarray(frame)`; one video item `fps = cfg.jev_sample_fps`; processor call kwargs `videos=[frames_pil]`, `video_metadata=[VideoMetadata(total_num_frames=n, fps=fps, frames_indices=list(range(n)), duration=n/fps)]`, `videos_kwargs={"do_sample_frames": False, "cap_pixels_per_frame": True}`.
3. **State (verbatim from the supplied `hazard_monitor.py`):** `{"task": "Warehouse CCTV safety monitoring. Judge the clip.", "camera": "cam-1", "clip": "<video:1>"}`, rendered by `render_state` (JSON, indent 2) inside `<state>…</state>`.
4. **Per hazard `h` (from `qev/prompt.py`, defaults, `PROMPT_VERSION = "qev-labels-v2"`, tagged layout):**
   - system turn: `SYSTEM_PROMPT` ("Apply the question to the state. Choose exactly one of the listed options. Respond with only its uppercase letter, with no explanation or reasoning.").
   - user turn: `<state>` block + suffix:
     ```
     Question: Is the statement true, or is the answer to the question yes?

     Options:
     A. yes: {HAZARD_QUESTIONS[h]}
     B. no: the statement is false / the answer is no

     Answer with one letter: A, B.
     ```
   - user content carries the video part (`{"type": "video"}`) at the `<video:1>` placeholder; the prompt text is produced via `apply_chat_template(..., tokenize=False, add_generation_prompt=True)` with content parts, then the processor expands the video pad.
5. **Readout (from `qev/mm_engine.py`/`engine.py`):** one forward pass with `logits_to_keep=1`; `slot_logits = logits[0, -1][[id_a, id_b]]`; `p_yes = softmax(slot_logits, temperature=1.0)[0]` (`qev/calibrate.softmax`; default `Calibration()` → T = 1.0); clamp to [0, 1] (`answers.make_noul`). No generation tokens are ever produced.
6. **Return** `{h: p_yes for h in HAZARD_QUESTIONS}` — all five questions per call. `validate_probs` (§7.3) re-checks the shape afterwards.
7. **Cost model:** five forward passes per tick (one per hazard; the shared-prefix/fork optimisations of `qev` are deliberately not replicated). Each pass prefills the same ≤12 frames (3 s window @ 4 fps) on a 0.8B model. Everything is inside `jev_time_s` (§5.2), so the benchmark and the UI see the true overhead. If it ever matters, prefix sharing is a Stage 2 optimisation — not now (§0.2 rule 8).
8. **No timeout, no retry, no caching** (§10.1, C-5).

## Appendix B — G1 analysis procedure (D-13: synthetic media)

**Inputs:** the GateLog JSONL of a gated run over a fully-decoding synthetic video (e.g. `tests.support.make_test_video` output — `synthetic_test.mp4` loses one frame at 12000 ms, F-16, which would blur `frames_saved_pct`) with `MockJevDecider()` (`DEMO_RULES`), plus `DEMO_RULES` itself as the pseudo-ground-truth. On synthetic media the "truth" is the mock rule table, so every check below is mechanically decidable.

**Script** (delivered in Phase 5 as `scripts/g1_report.py`, reads only the JSONL; runs on CPU):

1. **Reasons histogram** — counts per reason; every chunk accounted for.
2. **Frame economy** — Σ`frames_sent` / Σ`frames_full_equiv`, `frames_saved_pct`; per-tier ingest comparison when G0 data is present.
3. **Overhead ratio** — Σ`jev_time_s` / Σ`vlm_time_s`, and net saving = Σ(`frames_full_equiv` − `frames_sent`) chunk VLM-time estimate vs Σ`jev_time_s`.
4. **Window correctness (pseudo-GT):** every tick with `t_ms ∈ [10000, 12000]` must report `no_ppe ≥ route_full_threshold` (→ its chunk is `above_full`); every tick with `t_ms ∈ [5500, 8000]` must report `no_ppe ≥ route_watch_threshold` (→ at least `above_watch`).
5. **No violation chunk down-rated (target 0):** no chunk overlapping `[10000, 12000]` may be `REDUCED`.
6. **Early warning (C-9):** the first `above_watch`/`rising` chunk must precede the first `above_full` chunk (the 0.25 rule window ends at 8000, the 0.60 window starts at 10000).

**Real footage:** the same script runs unchanged on a real GateLog when the developer supplies footage; check 4–6 then need real GT labels and are reported as a table, not asserted (D-13).
