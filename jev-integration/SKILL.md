---
name: jev-integration
description: Guides an AI coding agent through the JEV v3 integration (design_v3.md "Domain Calibration & Gate v2") into the UI_TEST HERMES VLM harness — fp32 slot readout, shared-prefix OneJevDecider, iSafetyBench threshold calibration (calibrate_report/isafety_harness), GateRouter v2 (streak/hysteresis/EWMA-log), optional Phase E VLM track — with strict minimal additions-only changes, an additive regression suite, and the GPU handoff/permission protocol. Use whenever the task touches JEV, OneJev, gate routing, thresholds/calibration, iSafetyBench, GatedBatchWorker, jev_gate/frame_sampler/jev_adapter/gate_log, tick evaluation, route tiers, the gate JSONL log, compare_latency, or phrases like "continue the integration", "next phase", "verify the JEV work", "implement phase N", "calibrate JEV".
---

# JEV Integration (UI_TEST) — v3, Domain Calibration & Gate v2

## Mission
Implement `references/design_v3.md` exactly as written: smallest possible additions-only diff, an **additive regression suite** that only grows, **no regression** in detection quality or latency at any phase boundary (D-27), and every GPU/model/network job handed off to the developer (or run only with explicit per-run authorisation) until the developer clears it (D-32). The repo is a measurement harness — results must stay trustworthy; every claim of working is backed by pasted evidence.

## Read before acting (in full)
1. `references/design_v3.md` — the contract: normative logic (§5), data model (§6), change budget (§7), CLI contracts (§8), verification (§10), phases (§11), register (§12), schemas (Appendices B–D).
2. `references/verification_v3.md` — environment record, working loop, GPU handoff protocol, regression commands, diff/T-1 review, report templates, DEV handoff catalogue.
3. Historical: `references/design.md` (v2) for baseline facts, invariants and traps; `references/sources/qev/` — the official engine whose fp32 head and shared-prefix paths are being ported. Copy from sources, never from memory.

## Fixed environment facts
- **Repo:** `C:\Users\Maha Mamdouh\Desktop\New folder (3)\New folder\UI_TEST` (quote — spaces). Baseline `722cbc7` ("updating the prompts"); expected dirty set at session start: modified `calibrate_jev.py`, `compare_latency.py`, untracked `jev-integration/`. P0 normalises the tree; if the state differs, stop and report.
- **Python:** `F:/URCA_PROJECTS/.venv/Scripts/python.exe` (3.13.5, transformers 5.17.0, torch 2.14.0+cu126, CUDA available).
- **Model caches:** `HF_HOME=F:\URCA_PROJECTS\.hf_cache` — OneJev-0.8B snapshot already present; iSafetyBench (`raiyaanabdullah/isafety-bench`) downloaded in C0 (the pre-approved network job, D-31).
- Execution order: **P0 → A → B → C0 → C → D → E(optional)** — see design §4 for why this differs from alphabetical order.

## The minimalist contract
- **Change budget (design §7) is exhaustive.** New files and exact per-file edits only; the single permitted `gated_batch_worker.py` line is `ewma=decision.ewma,`. Anything else — including one-line "improvements" — is a stop-and-ask.
- **Never modify:** `temper/**`, `vlm/**`, all widgets, `config.py`, `chunker.py`, `batch_worker.py`, `player_worker.py`, `exporter.py`, existing tests, Dockerfile, docker-compose, requirements. No reformatting, renaming, import reordering.
- **Pinned values are sacred:** every name, signature, default, threshold, format string and expected test value in design_v3 (including Appendix D). Wrong-looking value → report, never edit.
- **Default path is sacred:** with no decider supplied, behaviour stays byte-identical; router v2 knobs are default-inert, proven by the frozen-v2 identity test.

## Hard rules
1. **GPU/network permission protocol (D-32):** never launch GPU, model-loading or network jobs. Default = prepare exact command + expected-output checklist + report NOT VERIFIED. Run one yourself only when the developer explicitly authorises that specific command; results count only when the developer clears them. Never upgrade NOT VERIFIED on your own judgement, never substitute mocks or CPU fallbacks for a GPU item.
2. **No timeouts, ever, in product code** (C-5): the words `timeout`, `QTimer`, `wait_for`, `alarm(`, `deadline`, `setTimeout` must not appear in new files or added diff lines (T-1 scan before every report). Shell `timeout` for your own test runs is fine.
3. **Lazy heavy imports (I-5):** torch/transformers only inside `OneJevDecider`; `--jev off`/mock paths never import them (JEV-8 after every phase).
4. **Additive regression suite (D-27):** the suite is green before the phase's first edit and after the last; a test that enters `vlm_test_ui/tests/` is never deleted, weakened, or skipped. Fail-on-break proof for every new test.
5. **Tests before the code they cover; stop after every phase (D-15)** — deliver the report (verification §6.1) and wait for the explicit go.
6. **JEV results never enter `session.events`**; gate data uses `gate_records`/`GateRecord` only (v2 trap 4).
7. **Calibration discipline:** thresholds live in `thresholds.json` until C's adoption gate passes; legacy points (0.20/0.35, 0.78/0.82) are reported, never trusted (D-30); any prompt edit voids the prompt hash and requires recalibration (D-28).

## Working loop (per phase)
verification_v3.md §2 verbatim: plan → tree check → regression suite BEFORE → implement minimal → fail-on-break proof → verify (tests twice, diff review, T-1) → DEV handoff → ledger row → report → STOP.

## Verified traps (each cost real time)
1. **bf16 slot-logit ties:** identical pmax across different chunks (F-21) is the fp32-head defect — Phase A exists to kill it; never "explain it away".
2. **RGB vs BGR:** harness frames are RGB; cv2 encode/decode is BGR. Tick frames read BGR, converted only for the decider; thumbnails from BGR.
3. **Chunk-id collisions:** `add_event` replaces same-`chunk_id` — keep JEV data out of `session.events`.
4. **Modal dialogs hang headless runs:** never `load_media` a missing path in tests.
5. **HERMES `conv_history` cadence:** every chunk is answered — never gate or skip generations.
6. **Tracked telemetry JSONs** at repo root are overwritten by benchmark runs — run benchmarks from a scratch directory.
7. **`compare_latency.py` path eviction:** workers purge the other tree; keep the pattern intact when editing.
8. **`sample_video_chunks` pre-resizes to 448**; the app does not — keep FULL chunks bit-identical to the hermes engine.
9. **Windows paths with spaces** — always quote.
10. **Prefix split:** the video placeholder must stay inside the prefix render; a mismatch raises `prompt prefix mismatch` → tick error → fail-open FULL. Position capture uses the decoder-layer hook (design §5.2 step 3), and the suffix continues at last-prefix-position + 1 on all three axes.
11. **`g1_report.py` reads `"type"`** after P0 — do not reintroduce `kind`; thresholds come from `run_start.gate_config` after C.
12. **argparse:** the master and worker of `compare_latency.py` must both define every flag they pass (F-23 was a real crash).

## Stop conditions — halt and ask when
- tree/commit state differs from expectation, or a cleared fact no longer holds;
- the change would exceed the §7 budget or touch a forbidden file;
- a test fails and the cause is unclear, or a DEV item's cleared output violates its gate (e.g. PREFIX-eq > 5e-3);
- the design is silent, ambiguous, or self-contradicting;
- the developer's latest message conflicts with the design (the developer wins; propose the register update — never edit §12 yourself unless asked).

## Tone
Plain, specific, short. Lead with what changed and whether it is verified. List what you did not do and why. No celebration, no hedging, no hiding failures.
