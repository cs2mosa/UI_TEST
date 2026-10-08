# Verification Reference — JEV integration v2

Contents: 1 Environment record · 2 Baseline · 3 Writing tests · 4 Diff review · 5 Screenshots · 6 What the agent cannot verify · 7 Report templates

A note on the word "timeout": the developer's *no-timeout* rule (design §10.1) governs **product code**. Using the shell's `timeout` command to stop *your own* test run from hanging your terminal is allowed and expected; it never ships.

All commands run with the project venv interpreter `F:/URCA_PROJECTS/.venv/Scripts/python.exe` (design D-22). Quote every output you cite.

---

## 1. Environment record (Phase 0, before anything else)
Run and paste the output. Do not infer the environment from the Dockerfile or requirements.

```bash
REPO="C:/Users/Maha Mamdouh/Desktop/New folder (3)/New folder/UI_TEST"   # quote — spaces in path
cd "$REPO"
git log -1 --format='%H %s'    # expect fedfb07 "second try"
git status --short             # expect empty (clean)
PY="F:/URCA_PROJECTS/.venv/Scripts/python.exe"
"$PY" --version                                                        # expect 3.13.5
"$PY" -c "import PySide6, cv2, numpy; print(PySide6.__version__, cv2.__version__, numpy.__version__)"
"$PY" -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect 2.14.0+cu126 True
"$PY" -c "import transformers; from transformers import Qwen3_5ForConditionalGeneration; print('transformers', transformers.__version__, 'Qwen3_5 OK')"  # expect 5.17.0
nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | tail -2
# OneJev-0.8B snapshot (D-19; download is the pre-approved Phase 0 action):
HF_HOME="F:\URCA_PROJECTS\.hf_cache" "$PY" - <<'EOF'
from huggingface_hub import snapshot_download
import os
p = snapshot_download('OmniJev/OneJev-0.8B')
print('SNAPSHOT:', p)
assert os.path.isfile(os.path.join(p, 'config.json'))
EOF
```
Record the printed snapshot path verbatim: it becomes the pinned `--jev-model-path` default (design §8.2) and is quoted in the Phase 0 report. If the tree is not at `fedfb07`/clean, **stop and report** — the design's facts were verified against that commit.

## 2. Baseline (record BEFORE any change; re-run after every phase)

Run from `vlm_test_ui/`, with `QT_QPA_PLATFORM=offscreen` on a machine with no display.

```bash
export QT_QPA_PLATFORM=offscreen
cd "$REPO/vlm_test_ui"
# R-1: the headless-runnable existing tests (expected: 11 passed)
timeout 120 "$PY" -m pytest tests/test_core.py tests/test_gt.py -v -p no:cacheprovider
# The pipeline test hangs where F:\URCA_PROJECTS does not exist (modal dialog). Do not edit it. Record it as:
#   "tests/test_pipeline.py::test_state_flow — not runnable in this environment (hardcoded F: path)"
```

R-2 / R-3 — save as `/tmp/baseline_flow.py` (outside the repo) and run from `vlm_test_ui/`:

```python
import os, sys
sys.path.insert(0, os.getcwd())
from PySide6.QtWidgets import QApplication
app = QApplication([])
from app.main_window import MainWindow
from app.config import DEFAULT_CONFIG
from examples.mock_vlm import MockVLM

path = os.path.abspath("examples/synthetic_test.mp4")
w = MainWindow(vlm=MockVLM(simulated_delay_s=0.01), model_label="baseline", config=DEFAULT_CONFIG)
states = [w.current_state]
w.load_media(path); states.append(w.current_state)
w._start_processing(); states.append(w.current_state)
w.batch_worker.wait(); app.processEvents(); states.append(w.current_state)
ev = [(e.chunk_id, e.start_ms, e.end_ms, e.num_frames, e.status) for e in w.session.events]
assert states == ["EMPTY", "LOADED", "PROCESSING", "READY"], states
assert ev == [(0, 0, 5000, 20, "ok"), (1, 4000, 9000, 20, "ok"), (2, 8000, 12000, 15, "ok")], ev
assert w.session.gate_records == [] and w.gate_panel is None          # JEV off: no gate artefacts
w.close()
print("BASELINE R-2/R-3 OK")
```
Expected last line: `BASELINE R-2/R-3 OK`. The line `This plugin does not support propagateSizeHints()` is harmless offscreen noise.

Never call `load_media` with a missing path in a test or script: it opens a modal `QMessageBox` and the run hangs. Patch `QMessageBox` if a test must exercise that path.

## 3. Writing tests
- **Style:** `unittest.TestCase` classes under `vlm_test_ui/tests/`, runnable by pytest, importing like the existing tests (`from app.core... import ...`).
- **No absolute paths, no network, no GPU, no real models** in any test you add. Locate media relative to the test file: `os.path.join(os.path.dirname(__file__), "..", "examples", "synthetic_test.mp4")` or build one with `tests.support.make_test_video` (tempfile).
- **Pure logic first.** `jev_gate`, `frame_sampler`, `gate_log` import neither Qt nor torch; test them without a `QApplication`.
- **Time:** never rely on real sleeps for correctness; `MockJevDecider(delay_s=...)` is the only sanctioned wait, and only to prove `jev_time_s` accounting. One separate real-clock pacing test is unnecessary in Stage 1 (no paced reader).
- **Threads:** a threaded test passes only if it passes **twice in a row** and when run alone. Report both runs.
- **Prove the test can fail** (design §12.2): for each new test, show once that it fails when the behaviour it guards is broken (e.g. drop the BGR→RGB conversion and watch the colour test fail), then restore. Paste both outputs.
- **Equivalence tests compare against the existing function** (`compute_chunk_windows`, `BatchWorker._sample_frames_for_window`), never against numbers typed by hand.
- **Lazy-import audit (JEV-8):** in the R-3/mock runs, assert `"torch" not in sys.modules and "transformers" not in sys.modules` afterwards.

## 4. Diff review (before every report)

```bash
cd "$REPO"
git add -N .
git diff --stat fedfb07
git diff --numstat fedfb07 | awk '$2 > 0 {print "DELETIONS:", $0}'   # each hit must match an edit the design allows (§7.2); list with justification
git diff --name-only fedfb07 | grep -E '^(temper/|vlm_test_ui/vlm/|vlm_test_ui/app/core/(vlm_adapter|chunker)|vlm_test_ui/app/workers/(batch|player)_worker|vlm_test_ui/app/io/exporter|vlm_test_ui/app/config|vlm_test_ui/app/widgets/(video_view|settings_bar|event_feed|event_card|chunk_timeline)|vlm_test_ui/tests/(test_core|test_gt|test_hermes.*|test_pipeline)\.py|Dockerfile|docker-compose)' \
  && echo "FORBIDDEN FILE TOUCHED" || echo "no forbidden files touched"
# T-1 no-timeout scan (design §10.1) over new files and added diff lines; every hit needs an explanation or removal
grep -rnEi 'timeout|QTimer|wait_for|alarm\(|deadline|setTimeout' vlm_test_ui/app/core/jev_gate.py vlm_test_ui/app/core/frame_sampler.py vlm_test_ui/app/core/jev_adapter.py vlm_test_ui/app/core/gate_log.py vlm_test_ui/app/workers/gated_batch_worker.py vlm_test_ui/app/widgets/gate_panel.py vlm_test_ui/examples/mock_jev.py vlm_test_ui/tests/support.py vlm_test_ui/tests/test_gate*.py vlm_test_ui/tests/test_jev_adapter.py vlm_test_ui/tests/test_compare_latency_3way.py vlm_test_ui/main.py scripts/g1_report.py 2>/dev/null || echo "T-1: no hits in new files"
git diff -U0 fedfb07 | grep -E '^\+' | grep -nEi 'timeout|QTimer|wait_for|alarm\(|deadline|setTimeout' || echo "T-1: no hits in diff"
```
Also run `git status --short` and account for **every** file. A reviewer must be able to read from your report: no reformatted lines, no renamed symbols, no reordered imports, no new dependency (I-5), default behaviour unchanged (R-1..R-3 re-run), and each changed existing function listed before/after.

## 5. Screenshots (UI-1) — evidence for a visual UI (Phase 4)
```python
w.grab().save("/tmp/ui_<state>.png")   # QWidget.grab works with QT_QPA_PLATFORM=offscreen
```
Capture EMPTY, LOADED, PROCESSING (mid-run, after at least one GateRecord and one VLM event), READY — each with JEV off and on (mock decider). Show them to the developer and describe what each shows. Do not describe a UI you have not rendered.

## 6. What the agent cannot verify — say so, and hand over the command
| Item | Why | Developer command |
| --- | --- | --- |
| ONEJEV-1 contract test (DEV-2) | needs CUDA + the downloaded snapshot | `QT_QPA_PLATFORM=offscreen "$PY" -m pytest vlm_test_ui/tests/test_jev_adapter.py -k onejev -v` |
| Real-model UI run (DEV-1) + G0 | needs GPU + HERMES + OneJev | `cd vlm_test_ui && ../.venv/Scripts/python.exe main.py --hermes --jev onejev` (then a second run with `--jev off` for the G0 baseline); VRAM peak via `nvidia-smi` |
| G1 analysis | runs on DEV-1's GateLog | `"F:/URCA_PROJECTS/.venv/Scripts/python.exe" scripts/g1_report.py <gate_log.jsonl>` |
| 60-min soak (DEV-3) | needs GPU + time | same entry point with a 60-min file; report memory, `wall_done_s`, JSONL line count |
| Three-way benchmark (DEV-4) | needs GPU + models | `cd <scratch dir> && "$PY" "C:/Users/Maha Mamdouh/Desktop/New folder (3)/New folder/UI_TEST/compare_latency.py" --video <file> --num-chunks 5` (scratch dir — tracked telemetry JSONs would be overwritten, F-14) |
| `test_pipeline.py::test_state_flow` | hardcoded `F:` path | `python -m pytest vlm_test_ui/tests/test_pipeline.py` on the Windows machine, before and after |

Write these as **NOT VERIFIED** in the report. Never convert "not verified" into "should work".

## 7. Report templates

### 7.1 Phase report
```
PHASE <n> REPORT — <name>
Approved decisions relied on: D-x (quote or date), ...
Files added: ...     Files changed: ... (each with lines added/deleted)
Baseline before: <pasted>      Baseline after: <pasted>
New tests: <ids> — run 1: <pasted>  run 2: <pasted>  fail-on-break proof: <pasted>
Diff review: <stat, deletions list with justification, forbidden-file check, T-1 output>
Screenshots: <paths + what they show>   (UI phases)
NOT VERIFIED (and why / developer command): ...
Deviations from the design: none | <list, each needing approval>
Open questions for the developer: ...
Stopping here for go-ahead (D-15).
```

### 7.2 Ask (blocked) report
```
I need a decision before continuing — <topic>
What I was doing: ...
Why I cannot decide: it is not in the Decision Register (§14) / the developer's instructions / the code.
Options:
  1. <option> — consequence ...
  2. <option> — consequence ...
My recommendation (not acted on): ...
What I will do once you answer: ...
What I am doing meanwhile: nothing on this branch | <independent, already-approved work>
```
