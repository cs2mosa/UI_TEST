# Verification & Execution Plan — JEV v3 (Domain Calibration & Gate v2)

Companion to `design_v3.md`. Contents: 1 Environment record · 2 The working loop · 3 GPU/network handoff protocol (D-32) · 4 Regression protocol · 5 Diff review & T-1 · 6 Report templates · 7 Metrics ledger · 8 Ask template · 9 DEV item handoff catalogue.

All commands use the project venv `F:/URCA_PROJECTS/.venv/Scripts/python.exe` (v2 D-22) and quote the repo path (spaces). **Quote every output you cite. "Should work" is not evidence.**

---

## 1. Environment record (run at session start, paste into the report)

```bash
REPO="C:/Users/Maha Mamdouh/Desktop/New folder (3)/New folder/UI_TEST"
cd "$REPO"
git log -1 --format='%H %s'     # expect 722cbc7 "updating the prompts"
git status --short              # expect: M calibrate_jev.py / M compare_latency.py / ?? jev-integration/
PY="F:/URCA_PROJECTS/.venv/Scripts/python.exe"
"$PY" --version                 # expect 3.13.5
"$PY" -c "import PySide6, cv2, numpy; print(PySide6.__version__, cv2.__version__, numpy.__version__)"
"$PY" -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.14.0+cu126 True (record only)
```
If `git status --short` differs from the expected dirty set: **stop and report** (design §11 P0). Phase P0 then asks the developer to commit the two modified files; after P0 the tree is expected **clean** (or exactly `?? jev-integration/` if the developer prefers to keep the package untracked) — record which, and verify at every later phase start.

---

## 2. The working loop (every phase; do not skip steps)

1. **Plan:** list the phase's files (design §7), its new tests (§10.1), the Approved decisions relied on (§12). Confirm nothing is [OPEN] and no register row is Pending.
2. **Tree check:** environment record; `git status --short` matches the expected state.
3. **Regression suite BEFORE:** run §4 agent suite; must be green. Paste output. (First session only: also capture the v2 R-2/R-3 script output as the flow baseline.)
4. **Implement minimal:** exactly the phase's §7 budget. Tests first or alongside. Match surrounding style (`from __future__ import annotations`, dataclasses, Qt signals). No reformatting of untouched lines.
5. **Fail-on-break proof:** for each new test, break the guarded behaviour once, show the test fail, restore. Paste both outputs.
6. **Verify:** new tests twice (threaded tests also run alone); full suite green; diff review + T-1 (§5).
7. **DEV handoff:** every GPU/model/network item → §3 protocol; list in the report.
8. **Ledger:** append the phase row to `references/metrics_ledger.md` (§7) with every metric you can compute agent-side; DEV-gated metrics stay `pending` until cleared.
9. **Report** (§6.1) and **STOP** for the developer's go (v2 D-15).

Never begin a later phase because an earlier one "looks done". If the developer names a phase, verify from the repo + ledger that all earlier phases are complete and their DEV items cleared; if not, say where the evidence ends.

---

## 3. GPU / network / model handoff protocol (D-32)

1. The agent never runs GPU, model-loading, or network jobs. There are exactly two sanctioned paths:
   - **Handoff (default):** the item is listed in the phase report as NOT VERIFIED with (a) the exact command, (b) the expected-output checklist (what to look at, what values gate what), (c) where the output lands. The developer runs it, pastes/clears the results; the next phase may only start once its dependency DEV items are cleared.
   - **Authorised agent run (exception):** the developer explicitly authorises a specific command in a message ("you may run X"). The agent runs exactly that command — nothing else — pastes the full output, and marks the item `VERIFIED (agent-run, developer-authorised)`.
2. The agent never substitutes a mock, a smaller input, or a CPU fallback to "get an answer": a GPU item that cannot run stays NOT VERIFIED (v2 D-21 spirit).
3. Never convert NOT VERIFIED into VERIFIED on your own judgement; only the developer's pasted output or explicit authorisation upgrades an item.
4. A cleared DEV item's output is quoted verbatim in the ledger row (or referenced to the pasted report section).

---

## 4. Regression protocol (D-27 — run before and after every phase)

**Agent part (CPU; every phase):**
```bash
export QT_QPA_PLATFORM=offscreen
cd "$REPO/vlm_test_ui"
# full suite (additive: it only grows; a red suite is a stop condition)
timeout 300 "$PY" -m pytest tests/ -v -p no:cacheprovider
# twice-in-a-row for the threaded gate tests
timeout 300 "$PY" -m pytest tests/test_gated_worker.py -v -p no:cacheprovider
timeout 300 "$PY" -m pytest tests/test_gated_worker.py -v -p no:cacheprovider
# flow baseline (v2 R-2/R-3 script; /tmp/baseline_flow.py from v2 verification §2, unchanged)
timeout 120 "$PY" /tmp/baseline_flow.py
```
`tests/test_pipeline.py::test_state_flow` is excluded as in v2 R-1 (hard-coded F: path, modal hang) — record it as not runnable here.

**Developer part (DEV anchors; handoff per §3):**
```bash
# latency anchor — 3-chunk jev_hermes worker only, scratch directory (tracked telemetry JSONs, v2 F-14)
cd <scratch> && "$PY" "C:/Users/Maha Mamdouh/Desktop/New folder (3)/New folder/UI_TEST/compare_latency.py" \
  --mode worker --engine jev_hermes --video <CLIP> --num-chunks 3 \
  --output anchor_<phase>.json
# quality anchor (Phase C onward)
"$PY" scripts/calibrate_report.py labeled_scores.jsonl --report-only thresholds.json
```
Compare each anchor to the previous phase's cleared numbers: M4 tolerance ±10 % unless the phase gate says otherwise; HERMES stage times must stay inside the band (the anchor shares the GPU with nothing else — note any background job in the report).

**Additive policy:** once a test file/function enters `tests/`, it is never deleted, weakened, renamed, or skipped. If a new legitimate behaviour changes an old expectation, the change is a register question first, and the old test gets a *new* case alongside, not an edit in place.

---

## 5. Diff review & T-1 (before every report)

```bash
cd "$REPO"
git diff --stat HEAD
git diff --numstat HEAD | awk '$2 > 0 {print "DELETIONS:", $0}'   # every hit must match §7 exactly
git status --short                                                  # account for every file
# forbidden-file check (v3 list; note gated_batch_worker.py allows exactly the one listed line)
git diff -U0 HEAD -- app/workers/gated_batch_worker.py | grep -E '^[+-]' | grep -v 'ewma=decision.ewma' \
  && echo "WORKER EDIT OUTSIDE BUDGET" || echo "worker edit within budget"
# T-1 no-timeout scan (v2 §10.1 word list) over new files and added lines
grep -rnEi 'timeout|QTimer|wait_for|alarm\(|deadline|setTimeout' \
  isafety_harness.py isafety_loader.py vlm_test_ui/scripts/calibrate_report.py \
  vlm_test_ui/tests/test_router_v2.py vlm_test_ui/tests/test_jev_adapter_v2.py \
  vlm_test_ui/tests/test_calibrate_report.py 2>/dev/null || echo "T-1: no hits in new files"
git diff -U0 HEAD | grep -E '^\+' | grep -nEi 'timeout|QTimer|wait_for|alarm\(|deadline|setTimeout' \
  || echo "T-1: no hits in diff"
```

---

## 6. Report templates

### 6.1 Phase report (v3 edition)
```
PHASE <P0|A|B|C0|C|D|E> REPORT — <name>
Approved decisions relied on: D-xx (quote/date), ...
Files added: ...    Files changed: ... (each with +/- lines and the §7 budget line it implements)
Environment record: <pasted>
Regression suite BEFORE: <pasted last lines>      AFTER: <pasted last lines>  (threaded test runs included)
New tests: <ids> — run 1 <pasted> run 2 <pasted> fail-on-break proof <pasted>
Diff review: <stat, deletions w/ justification, worker-edit check, T-1 output>
DEV handoff table: item | command | expected-output checklist | status (NOT VERIFIED / CLEARED by dev / VERIFIED agent-run authorised)
Metrics ledger row appended: <row>
Baseline/anchor deltas: <agent-side metrics; DEV anchors pending/cleared>
Deviations from the design: none | <list, each needing approval>
Open questions: ...
Stopping here for go-ahead (D-15).
```

### 6.2 DEV item clearance (developer pastes output; agent records)
```
DEV <id> CLEARED: <one-line result> — gates <pass/fail>: <numbers>
```

---

## 7. Metrics ledger — `jev-integration/references/metrics_ledger.md`

Append-only; one row per phase; `pending` until the DEV item clears; never rewrite history.

```markdown
| Phase | Date | M1 recall | M2 precision | M3 saved % | M4 jev s/tick | M5 early-warn | M6 HERMES | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 2026-10-.. | n/a | n/a | n/a | 28.5 (scratch, uncleared) | n/a | n/a | report §.. |
```
(M4 baseline from `scratch_jev_worker_test.json`: 129.4 s and 182.0 s per chunk across ~5–6 ticks ≈ 25–36 s/tick — record the developer's anchor run as the authoritative baseline.)

---

## 8. Ask (blocked) template — v2 §7.2 verbatim
```
I need a decision before continuing — <topic>
What I was doing: ...
Why I cannot decide: register / developer instructions / code are silent or conflict.
Options: 1. <option> — consequence  2. <option> — consequence
My recommendation (not acted on): ...
What I will do once you answer: ...
What I am doing meanwhile: nothing on this branch | <independent already-approved work>
```

---

## 9. DEV item handoff catalogue (fills as phases progress)

| ID | Phase | Command | Expected-output checklist |
| --- | --- | --- | --- |
| DEV-A1 (PROBE-1) | A | `"F:/URCA_PROJECTS/.venv/Scripts/python.exe" isafety_harness.py --dataset-dir <ISAFETY_SNAPSHOT> --mapping isafety_mapping.json --model-path <ONEJEV_SNAPSHOT> --limit 20 --out probe_a.jsonl --dump-logits probe_a_logits.jsonl` | run_meta shows `head_dtype: "float32"`; logits file: no two distinct items share bit-identical (logit_a, logit_b); step-size histogram < 0.125 resolution; per-hazard p_yes distributions before/after pasted |
| DEV-B2 (PREFIX-eq) | B | `"..python.exe" isafety_harness.py --dataset-dir <SNAP> --mapping isafety_mapping.json --limit 20 --out eq_b.jsonl --dump-logits eq_b_logits.jsonl` (run twice: once with `--decider-mode fullpass`, once shared — the harness grows a `--decider-mode {shared,fullpass}` flag in Phase B for exactly this test) | max abs diff of p_yes per (item, hazard) ≤ 5e-3; above → STOP, fast path not shipped |
| DEV-C1 (download, D-31) | C0 | `HF_HOME="F:\URCA_PROJECTS\.hf_cache" "$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('raiyaanabdullah/isafety-bench'))"` | printed snapshot path recorded; annotation files listed |
| DEV-C2 (scoring) | C | `"..python.exe" isafety_harness.py --dataset-dir <SNAP> --mapping isafety_mapping.json --out labeled_scores.jsonl --null-input --order both` | item counts per subset rule; zero skipped-or-accounted lines; run_meta hash present |
| DEV-C3 (adoption) | C | **Agent-runnable part:** `"..python.exe" scripts/calibrate_report.py labeled_scores.jsonl` (CPU) → exit 0 = adoption gates met; then the synthetic G1 rerun of design §10.1 (agent-runnable, threshold-aware). **DEV part (real-footage domain shift):** `cd <scratch> && "$PY" ".../UI_TEST/compare_latency.py" --mode worker --engine jev_hermes --video <REAL_CLIP> --num-chunks 5 --gate-config-json thresholds.json --output c3_real.json` | report exit 0; holdout recall ≥ 0.95, precision ≥ floor; real-footage run: no obviously-hazardous chunk left REDUCED (developer's judgement, recorded), M4 band respected, `jev_tier`/`jev_reason` present in telemetry |
| DEV-anchor | every | §4 latency anchor | M4 band respected; stage times stable |
| DEV-E-G0 | E | per-item G0 command (worker run with/without the E item enabled) | M6: no metric worse, ≥ 1 better |
| DEV-final | end | full master from scratch dir: `"$PY" compare_latency.py --video <CLIP> --num-chunks 5 --gate-config-json thresholds.json` | three engines complete; JEV+HERMES telemetry carries the adopted thresholds; ledger final row |
