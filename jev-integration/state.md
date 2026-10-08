# JEV Integration — Project State Log

Append-only chronological record of every evolving state of the JEV integration work, for full
context by any later agent or the developer. Newest entries at the bottom; correct forward, never
rewrite. Companion docs: `IMPLEMENTATION.md` (ideas + implementations), `PROCESS.md` (how we
worked), `references/design_v3.md` (contract), `references/metrics_ledger.md` (per-phase metrics).

---

## 2026-10-05 — v2 contract approved; v2 phases 0, 1, 2, 3, 6 shipped

- Decision Register D-1..D-23 confirmed with the developer (v2 design, `references/design.md`).
- **Implemented and committed** (see `git log`): Phase 0 (env, OneJev download, baseline),
  Phase 1 (`jev_gate.py`, `frame_sampler.py`, `TickRecord`/`GateRecord`), Phase 2
  (`jev_adapter.py`, `gate_log.py`, `mock_jev.py`, session-model additions), Phase 3
  (`GatedBatchWorker`), Phase 6 (`compare_latency.py` three-way: HERMES / InfiniPot-V /
  JEV+HERMES). Commit lineage: `7f7f10d` → `219fb1c` → `452644c` → `722cbc7`.
- **Deliberately NOT done:** v2 Phase 4 (GatePanel + `main.py`/`main_window.py` UI wiring — no
  `--jev` CLI flags in the app) and Phase 5 (real-model DEV runs). v3 keeps these open as v2
  work and runs the DEV vehicle through the CLI/worker path instead.
- Mock-only verification at this point; no real OneJev numbers yet.

## 2026-10-06 — reality check: scores untrustworthy, JEV too slow; v3 package written

- First real runs (`scratch_jev_worker_test.json`) exposed two blockers:
  - **F-21:** chunks 0 and 1 report bit-identical `jev_pmax = 0.4073334336280823` on different
    frames → bf16 slot-logit quantization (magnitude 25–50 → 0.125 steps → exact ties).
  - **Cost:** JEV = 129–182 s per chunk (five full forward passes per tick sharing the GPU with
    HERMES) ≈ 55–100 % of the HERMES chunk cost.
- Real OneJev `pmax` tops out ≈ 0.41, so the v2 default thresholds (0.20/0.35) routed everything
  FULL; an uncommitted recalibration moved the sweep to 0.78/0.82 in `calibrate_jev.py` and
  `compare_latency.py` (worker `getattr` fallbacks). Both pairs later demoted to "legacy
  candidates" (D-30) — reported, never trusted.
- Known-bug list recorded: F-23 (benchmark master passes undefined argparse flags → crash),
  F-24 (`g1_report.py` reads `kind`/`media_path`/`decider_label`/`ts` but the log writes
  `type`/`media`/`decider`/`time`).
- **v3 design package completed** ("Domain Calibration & Gate v2"):
  `references/design_v3.md` (v3.1, register D-24..D-33: iSafetyBench calibration, recall-first
  with precision floors 0.60/0.40, prompt-hash pinning, edit-budget amendment D-29, GPU
  permission protocol D-32, steering excluded D-33), `references/verification_v3.md` (working
  loop, DEV handoff catalogue), `SKILL.md` (v3 operating manual), `START_HERE.md` (start
  prompt). Execution order fixed: **P0 → A → B → C0 → C → D → E(optional)**.

## 2026-10-07 (session 1) — grounding, skill sync, pre-flight decisions

- Full re-ingestion of the repo + reference package; v3 baseline facts verified live (F-23/F-24
  present, F-21 in the scratch file, both model snapshots in `F:\URCA_PROJECTS\.hf_cache\hub`).
- **Installed skill copy was still v2** → synced the whole package to
  `~/.agents/skills/jev-integration/` (verified byte-identical via `diff -r`).
- Developer decisions: **D-34 (proposed)** — the two uncommitted files stay uncommitted until
  the end (commit once, under a green suite); 0.60/0.40 floors proceedable as CLI defaults.

## 2026-10-07 (session 2, main implementation day)

Developer instruction: work through the phases continuously, report at each boundary (D-15
amended). All phases below ran the full working loop (suite BEFORE/AFTER, fail-on-break proofs,
diff/T-1 review, ledger row, report).

- **P0 done.** F-23 fixed (flags defined, master passes only-when-set, jev_hermes-only
  rejection, worker builds GateConfig from set values → defaults 0.20/0.35 restored); F-24
  fixed (`type`/`media`/`decider`/`time`). Baseline latency anchor handed off as DEV item.
- **A done.** fp32 LM head ported verbatim (§5.1) into `OneJevDecider.__init__` (`head_dtype`
  attribute, "model" fallback + warning); `isafety_harness.py` created at repo root (scoring +
  `--dump-logits` probe via an LM-head forward hook — no prompt duplication; `prompt_hash`
  computed per the D-28 glossary formula, starts `4cd9341857fdc94f…`).
- **B done.** Shared-prefix tick evaluation (`MARKER` split, one prefill, position capture via
  decoder-layer hooks, five deepcopy'd suffix branches, `order` kwarg, `mode` dispatch with the
  v2 five-pass kept as `_call_fullpass`); `isafety_harness.py --decider-mode`; 11 unit tests
  with a hook-firing stub model (CPU-only); 4 fail-on-break proofs.
- **C0 (agent part) done.** Schema researched from the official repo (annotations =
  `{video_name, caption, gt_actions[]}`; exactly 98 routine + 67 hazard actions, matching F-26);
  `isafety_mapping.json` DRAFT (15/67 mapped; **`no_ppe` maps to NOTHING — the taxonomy has no
  PPE-violation action**; 52 unmapped with justifications); `isafety_loader.py` (loud schema
  validation, unknown-tag abort, strict-exclusion rule + loose-reading count reported);
  smoke-tested on a synthetic dataset dir. Mapping approval + snapshot validation → server.
- **C (agent part) done.** `scripts/calibrate_report.py` (§5.4 complete); 13 CAL tests pinning
  the Appendix D operative values (AUC 0.625; floor 0.60 → t=0.65) + 4 gate-config plumbing
  tests; `compare_latency.py --gate-config-json` (master fail-fast, worker pass-through,
  explicit-flags-override); `g1_report.py` check-4 reads thresholds from
  `run_start.gate_config`. Deviations disclosed: **Appendix D's inline precision labels for
  t=0.15/t=0.25 are swapped in the doc** (operative pins unaffected); **g1_report check-4
  `vec`→`probs`** (third latent F-24-family bug, invisible until P0 fixed the partition);
  LOG-4 expected-keys extended with `"ewma"` later in D (mandated by design §6).
- **Synthetic G1 rerun (DEV-C3 agent part) green:** 6/6 checks under current defaults,
  thresholds consumed from `run_start.gate_config`, 28.6 % frame savings, early warning
  precedes above_full.
- **D done.** GateRouter v2 (streak / hysteresis 3b / EWMA log-only / `prompt_hash`) with the
  pinned rule table; `RouteDecision.ewma` + `GateRecord.ewma`; `gate_record_to_dict` trailing
  `"ewma"`; `gated_batch_worker.py` exactly its one line; 14 ROUTE2/LOGX tests including the
  **frozen-v2 identity over 1000 seeded randomized sequences**; semantics discovery documented
  (streak builds via rules 4/5; rule 6 resets it). 3 fail-on-break proofs.
- **Suite state after all phases: 176 passed + 6 pre-existing `test_hermes_engine.py` GPU-load
  errors + 7 subtests** — identical error profile before and after every phase (the 6 errors
  are the environment issue below, present in the very first BEFORE run).

## 2026-10-07 (session 2, cont.) — compliance review closes the last gap

- Developer-requested requirement-by-requirement audit of design_v3 vs the tree: **one genuine
  miss found** — `isafety_harness.py` lacked `--null-input` and `--order both` (§5.3/§8).
  Implemented and smoke-verified with a hook-firing stub (null rows → null-input table with
  logit offsets; order-probe `ba` rows → order-bias table). The smoke caught a real bug in the
  first draft (`ba` rows carried the item's source instead of the pinned `source="order_probe"`)
  — fixed. T-1 clean; adjacent tests 56 passed.

## 2026-10-07 (session 2, cont.) — the GPU environment story

- **Authorized baseline anchor run (venv)** failed: HERMES 4-bit load crashes in accelerate
  dispatch — bitsandbytes `Tensor.item() cannot be called on meta tensors`
  (transformers 5.17.0 / accelerate 1.15.0 / bitsandbytes 0.50.2 / torch 2.14.0+cu126, Quadro
  RTX 3000). Identical root cause as the 6 pre-existing suite errors. OneJev loads fine.
- **Docker chapter** (developer's usual verification vehicle; agent chose build-and-test-now):
  - Review fixed the Dockerfile: **+`libegl1`** (PySide6 unimportable in container) and
    **+`pytest==8.4.2`** (image couldn't run the suite). Proposed (not applied):
    `docker/entrypoint.sh` `bash|sh)` should be `exec bash "$@"`.
  - Container suite (code volume-mounted): 167 passed, 10 skipped, 5 failures = all the
    `test_gated_worker` spin-budget artifact (60k `processEvents` too slow on the container
    event loop; direct diagnostic with 200k spins shows the worker completing with the
    identical routing summary). Not a code defect.
  - Disk-full episode killed a build and a probe (Docker engine dies with `unexpected EOF`);
    the space cleanup also removed the built image; rebuilt from scratch successfully
    (24.5 GB, includes both Dockerfile fixes).
  - **1-chunk GPU probe on the clean container → the meta-tensor crash REPRODUCED.**
    Conclusion: the crash is the pinned library trio itself, not a dirty venv. Untested
    hypothesis: 6 GB VRAM forces the crashing CPU-offload path — a bigger GPU may not crash,
    so **test as-is on the server before bumping versions**.
- **Developer policy (binding):** no expensive GPU tests locally — small tests only (syntax/
  logic bar); all runtime/model verification moves to the server. Docker chapter closed.

## Current state (snapshot, 2026-10-07 end of day)

- **Git:** HEAD `722cbc7` ("updating the prompts"); nothing committed from the integration.
  Dirty set: `M Dockerfile` (libegl1 + pytest), `M calibrate_jev.py` (developer's 0.78/0.82
  recalibration, pre-existing), `M compare_latency.py`, `M vlm_test_ui/app/core/jev_adapter.py`,
  `M vlm_test_ui/app/core/jev_gate.py`, `M vlm_test_ui/app/core/types.py`,
  `M vlm_test_ui/app/workers/gated_batch_worker.py`, `M vlm_test_ui/scripts/g1_report.py`,
  `M vlm_test_ui/tests/test_gate_log.py`; untracked: `isafety_harness.py`, `isafety_loader.py`,
  `isafety_mapping.json`, `jev-integration/`, `vlm_test_ui/scripts/calibrate_report.py`,
  `vlm_test_ui/tests/test_calibrate_report.py`, `vlm_test_ui/tests/test_jev_adapter_v2.py`,
  `vlm_test_ui/tests/test_router_v2.py`. Total diff vs HEAD: 9 files, +404/−58.
- **Suite:** 176 passed + 6 known GPU-environment errors + 7 subtests (see above).
- **Docker:** `urca-vlm:latest` built from the fixed Dockerfile; GPU passthrough verified;
  container suite verified (modulo the spin-budget artifact).
- **Ledger:** rows P0 → D appended in `references/metrics_ledger.md` (DEV-gated metrics
  `pending`).
- **Environment:** venv `F:/URCA_PROJECTS/.venv` (Python 3.13.5, torch 2.14.0+cu126 —
  HERMES 4-bit load BROKEN, see above); Docker image torch 2.14.1+cu130 — same crash; local
  GPU Quadro RTX 3000 6 GB; server is the GPU-heavy/CPU-small target (Dockerfile tuned for
  RTX 5090 / CUDA 13).

## Pending — everything left, in execution order (server unless noted)

1. **Server: run the stack as-is.** Build the image on the server GPU; run the three-way
   benchmark / 3-chunk latency anchor from a scratch dir. If the meta-tensor crash reproduces
   there → bump bitsandbytes/accelerate (developer decision; requirements/Dockerfile are
   theirs) and rebuild. If not → record the anchor as the P0/B baseline (M4).
2. **DEV-A1 (PROBE-1):** fp32 readout probe — `isafety_harness.py --dataset-dir <snapshot>
   --mapping isafety_mapping.json --limit 20 --out probe_a.jsonl --dump-logits
   probe_a_logits.jsonl`; expect `head_dtype:"float32"`, no bit-identical logits across items,
   sub-0.125 step resolution → confirms/refutes F-21 in the ledger.
3. **DEV-B2 (PREFIX-eq):** same command twice (`--decider-mode fullpass`, then `shared`) →
   `max |p_suffix − p_fivepass| ≤ 5e-3` per (item, hazard); above → fast path does not ship.
4. **DEV-C1 + C0 completion:** download `raiyaanabdullah/isafety-bench` into the HF cache;
   validate `isafety_loader.py` against the real snapshot layout (adjust the loader only if
   the layout differs from the documented one); **developer approves `isafety_mapping.json`**
   (especially the empty `no_ppe` mapping and the strict-vs-loose exclusion counts the loader
   prints).
5. **DEV-C2 + adoption:** full scoring run (`--null-input --order both`) →
   `calibrate_report.py labeled_scores.jsonl` → exit 0 (holdout recall ≥ 0.95, precision ≥
   floor) → `thresholds.json` written; check the `ADVISORY` line for legacy dominance.
6. **DEV-C3:** real-footage gated run with `--gate-config-json thresholds.json`; G1 analysis
   on the real log (checks 4–6 reported as a table there, not asserted).
7. **Final:** three-way benchmark from a scratch dir with the adopted thresholds; ledger final
   row; **developer commits everything** (D-34).
8. **Developer ratifications queued:** D-34 + D-15-amendment register entries; the four
   disclosed deviations (LOG-4 ewma-key extension; g1_report `vec`→`probs`; Appendix D label
   swap — doc-side; baseline-flow `not hasattr` adaptation); the flaky
   `TestWin1BaselineEquivalence` test (stabilise or accept); `entrypoint.sh` argument fix.
9. **Phase E** — optional, only after A–D are verified and running smoothly (D-26); not started.
