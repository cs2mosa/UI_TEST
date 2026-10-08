# JEV v3 Integration — Process Record

**Audience:** any coding agent (or developer) continuing this work. This file records **how the
project was actually run** — the working protocol, the decision-making rules, the verification
philosophy, the tooling traps, and the Docker chapter — so a newcomer can work the same way
without re-learning any of it. Companion files: `references/design_v3.md` (the contract),
`references/verification_v3.md` (the verification plan), `IMPLEMENTATION.md` (what was built
and why), `state.md` (chronological state log), `references/metrics_ledger.md` (per-phase
metrics), and `SKILL.md` (the condensed operating manual installed as a user-scope skill).

---

## 1. The document hierarchy (who wins when things conflict)

1. **The developer's latest message** — always wins. When it conflicts with the design, the
   agent follows the developer and *proposes a register entry* (the agent never edits §12 of the
   design doc itself unless asked).
2. **`references/design_v3.md`** — the contract: normative logic (§5), change budget (§7,
   exhaustive — anything not listed is forbidden), CLI contracts (§8), phases (§11), decision
   register (§12), pinned values (appendices). "Pinned values are sacred": a wrong-looking
   value is **reported, never edited** (this happened for real — see §5, Appendix D).
3. **`references/verification_v3.md`** — the working loop, GPU handoff protocol, regression
   commands, report templates, DEV handoff catalogue.
4. **`SKILL.md`** — the condensed operating manual; installed at user scope
   (`~/.agents/skills/jev-integration/`) and synced from this package whenever the package
   changes, so fresh agent sessions load the same rules automatically.
5. Historical: `references/design.md`/`verification.md` (v2) for baseline facts and traps.

Stop conditions (halt and ask): tree/commit state differs from expectation; a change would
exceed the §7 budget or touch a forbidden file; a test fails and the cause is unclear; the
design is silent, ambiguous, or self-contradicting; a DEV item's cleared output violates its
gate. Anything the design is silent on is **a question for the developer, not a decision**.

---

## 2. The working loop (used for every phase, verbatim from verification_v3 §2)

1. **Plan** — list the phase's files (§7), its new tests (§10.1), the Approved decisions relied
   on (§12). Confirm nothing is [OPEN].
2. **Tree check** — environment record; `git status --short` must match the expected state.
3. **Regression suite BEFORE** — must be green; paste the output.
4. **Implement minimal** — exactly the phase's §7 budget; tests first or alongside.
5. **Fail-on-break proof** — for each new test: break the guarded behaviour once, show the test
   fail, restore. (Actually done for every test family: e.g. corrupted midpoint candidates,
   mismatch check removed, positions without the +1, order mapping ignored, streak gate
   bypassed, EWMA blend inverted, hysteresis `ge`→`>`.)
6. **Verify** — new tests twice; full suite green; diff review (deletions must map to §7 lines)
   + T-1 no-timeout scan.
7. **DEV handoff** — every GPU/model/network item → protocol below.
8. **Ledger** — append the phase row to `references/metrics_ledger.md` (append-only, M1–M6,
   `pending` until DEV items clear).
9. **Report** (verification §6.1 template) and stop — see the D-15 amendment below.

**The regression suite is additive (D-27):** a test that enters `vlm_test_ui/tests/` is never
deleted, weakened, or skipped. When legitimate new behaviour invalidates an old expectation,
the change is a register question first. One real case: §6 of the design *mandates* appending
`"ewma"` to the gate-record dict key order, which broke the old exact-order assert (LOG-4) —
the minimal faithful change (append the key to the expected list, with a comment) was applied
and disclosed for ratification.

---

## 3. The GPU/network permission protocol (D-32) — and how it evolved

**Rule:** the agent never launches GPU, model-loading, or network jobs. Default = prepare the
exact command + an expected-output checklist and report the item **NOT VERIFIED**. Two sanctioned
exceptions: the developer runs it and clears the results, or the developer explicitly
authorizes that specific command in a message (then the agent runs exactly that, pastes the
output, marks `VERIFIED (agent-run, developer-authorised)`). Never substitute mocks, smaller
inputs, or CPU fallbacks for a GPU item.

**How it played out:** the baseline latency anchor was first handed off (NOT VERIFIED); the
developer then authorized it on a specific synthetic video (agent-run); it failed on the
HERMES 4-bit load → diagnosed (see IMPLEMENTATION §8) → later reproduced deliberately in the
clean Docker container to pin the root cause; finally the developer set the standing policy:

> **No expensive GPU tests locally. Small tests only — if no syntax/logic error, that's it.**
> All runtime/model verification (anchors, PROBE-1, PREFIX-eq, scoring, three-way benchmark)
> happens on the server.

That policy is now binding: local verification = the CPU regression suite + smoke tests; the
server runbook in `state.md` carries everything else.

**The agent-side vs DEV-side split in practice.** Everything expressible without a GPU/model/
network was implemented and verified by the agent (the full suite, stub-based model tests,
CPU report tooling, synthetic-media worker runs). Everything needing hardware or the dataset
is a DEV item with command + checklist in `verification_v3.md` §9. The split is by *evidence
type*, not by effort.

---

## 4. Decision-making record (beyond the design register)

These were developer decisions made mid-flight that shaped the work (design §12 holds D-1..D-33;
the agent proposed, the developer decided):

| Decision | What was decided |
| --- | --- |
| **D-15 amended (continuous work)** | "Work your way with the phases and report once each phase finishes" — the stop-and-wait after every phase became report-and-continue; the agent stops only where a DEV item physically blocks. |
| **D-34 (tree stays dirty)** | The pre-existing uncommitted threshold-recalibration changes in `calibrate_jev.py`/`compare_latency.py` stay uncommitted for the whole integration; the developer commits at the end, once everything is correct under the regression suite. Diff reviews attribute those hunks to the developer, never to budget violations. Expected dirty set at every phase start: `M calibrate_jev.py`, `M compare_latency.py`, `?? jev-integration/` + that phase's own edits. |
| **Dataset deferred** | iSafetyBench download (DEV-C1) postponed to the server (local storage); all calibration postponed with it. |
| **Internet schema allowed** | Where the dataset work doesn't need the download, researching the schema from official sources is sufficient. This unlocked C0's loader + mapping draft (schema validated loudly in code; real-snapshot validation happens on the server). |
| **GPU-first requirement** | "Every fix you made that works on CPU only will need to be able to run on GPU whenever GPU is present" — the server is GPU-heavy/CPU-small. Implementation answers: device strings derived from inputs, no CPU-fallbacks anywhere, CUDA enforced where the design says so. |
| **Docker verification chosen** | For the "verify now vs verify at the end" question the developer offered, the agent chose build-and-test-now (see §6). |
| **Small-tests-locally policy** | See §3 above. |

---

## 5. Verification philosophy (what counts as evidence)

- **"Should work" is not evidence.** Exact commands and pasted outputs for every claim; the
  metrics ledger records only what was measured, with `pending` for DEV-gated numbers.
- **Fail-on-break proofs**: every new test was shown to fail when the behaviour it guards was
  broken, then restored. This caught two real bugs during development (the thresholds.json
  holdout-key bug, the order-probe `source` bug) and surfaced one pre-existing doc bug
  (Appendix D label swap — reported, never edited).
- **Oracles over hand-typed values**: equivalence tests compare against the existing function
  (frozen v2 router, baseline sampler) or hand-computed design values (Appendix D), never
  against numbers an agent invented.
- **Small tests locally, runtime evidence on the server**: syntax/logic errors are the local
  bar; per-tick probability equivalence, fp32 readout resolution, latency, and detection
  quality are server evidence.

---

## 6. The Docker chapter (build-and-test-now)

The developer's usual verification vehicle is a Docker image (`Dockerfile` +
`docker-compose.yml` + `docker/entrypoint.sh`, NVIDIA CUDA 13 base, Python 3.13 venv, pinned
wheels). The agent chose to review, fix, build, and test locally rather than defer everything.

**Review findings (two fixed, one proposed):**
1. **`libegl1` missing** from the apt list — PySide6 could not even import in the container
   (`libEGL.so.1: cannot open shared object file`), killing every Qt test and UI service.
   Found via container suite collection errors. **Fixed** (one apt addition).
2. **`pytest` missing** from the pip list — the image couldn't run the repo's own test suite.
   **Fixed** (one pip addition).
3. **`entrypoint.sh` `bash|sh)` case discards arguments** (`exec bash` drops `-lc "..."`).
   Proposed fix `exec bash "$@"`; workaround used meanwhile: dispatch commands through the
   pass-through case (`CMD=python`/`nvidia-smi` directly).

**What the container verified:** GPU passthrough works (Quadro RTX 3000 visible via
`--gpus all`); the repo code can be volume-mounted over `/workspace` (so code changes don't
require a rebuild); the suite inside the container: 167 passed, 10 skipped (GPU tests without
`--gpus`), 5 failures = all the spin-budget artifact (IMPLEMENTATION §8.4).

**What the container decided:** the 1-chunk GPU anchor probe on the clean image reproduced the
HERMES 4-bit load crash → the pinned `transformers 5.17.0 + accelerate 1.15.0 + bitsandbytes
0.50.2` trio is the root cause, not a dirty venv. Because 6 GB VRAM may itself force the
crashing offload path, the server instruction is **try as-is first, bump versions only if it
reproduces there**.

**Operational notes for future agents (Windows + Git Bash + Docker Desktop):**
- Docker data lives on F: (C: is nearly full); a disk-full event kills the Docker engine
  mid-build/mid-run with cryptic `unexpected EOF` / `rpc error` messages — check `df -h` first
  when builds die mysteriously.
- Git Bash mangles absolute paths in `docker run` args → prefix `MSYS_NO_PATHCONV=1`.
- `docker build ... 2>&1 | tail -N` buffers all output until completion — pipe to a file or
  drop the tail if you need to poll progress.
- The entrypoint starts Xvfb and prints xkbcomp noise on stderr — filter it; its presence
  confirms the entrypoint ran.
- `docker compose run benchmark ...` expects the video under `/data` and writes telemetry to
  `/results`; the HF cache mounts from `${HF_CACHE_DIR:-./hf_cache}` — point it at
  `F:\URCA_PROJECTS\.hf_cache` so the models are found without downloads.

---

## 7. Tooling & environment traps learned the hard way (keep this list)

1. **Windows paths with spaces** — always quote (`C:/Users/.../New folder (3)/...`).
2. **QThread + Qt signals in tests**: `worker.wait()` blocks without processing events — queued
   signals never deliver. Use the `processEvents()` spin loop pattern (see
   `tests/test_gated_worker.py::_run_worker`). The spin budget (60k) is environment-sensitive:
   containers need more.
3. **`finished` is emitted twice** for workers (custom emit + native QThread teardown) — assert
   "all outcomes are finished", not "exactly one".
4. **Never `load_media` a missing path in tests** — it opens a modal dialog and hangs headless.
5. **Tracked telemetry JSONs** at the repo root are overwritten by benchmark runs — run
   benchmarks from a scratch directory (the compose file's `/results` mount exists for this).
6. **Event-loop differences**: container offscreen Qt is slower; timing-based budgets must be
   generous or (better) not exist.
7. **`git diff` on untracked files shows nothing** — account for new files via
   `git status --short`; the T-1 scan greps new files by name.
8. **CRLF warnings** on edited files are cosmetic (autocrlf); Python doesn't care.
9. **Background long jobs**: pipe output to a file (not `tail`) so progress is polluable; the
   harness that spawns them may lose them across session continuations — prefer checking
   `docker ps` / logs over trusting a silent notification trail.
10. **Append-only artifacts**: the metrics ledger and `state.md` are history — correct forward
    with new entries, never rewrite old ones.

---

## 8. If you are the next agent: where to start

1. Read `state.md` top-to-bottom (it ends at "Pending" — that's your backlog).
2. Read `references/design_v3.md` §7 (budget) and §12 (register) before touching anything.
3. Run the regression suite first (`verification_v3.md` §4); if it isn't green at ~176 passed
   (+6 known GPU-environment errors), stop and diff the tree against `state.md`'s snapshot.
4. Whatever you implement: suite before and after, fail-on-break proofs, T-1, ledger row,
   report — and when a GPU is involved, the handoff protocol of §3.
