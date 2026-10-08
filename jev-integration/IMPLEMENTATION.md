# JEV v3 Integration — Implementation Record

**Audience:** any coding agent (or developer) picking up this repo. Read this together with
`references/design_v3.md` (the contract), `references/verification_v3.md` (the verification plan),
`PROCESS.md` (how we worked), and `state.md` (chronological state log). This file records **what was
built, the idea behind each piece, and how it was implemented and verified** — everything an
implementing agent needs to avoid re-deriving or, worse, silently changing it.

**Status when this was written (7 Oct 2026):** all agent-side phases (P0, A, B, C0-agent-part,
C-agent-part, D) implemented, tested, and reported; nothing committed (developer decision D-34:
commit once at the end, under a green regression suite). Every remaining item is dataset- or
GPU-runtime-dependent and runs on the server — see `state.md` § "Pending".

---

## 0. The system in one page

UI_TEST is a measurement harness for a warehouse-CCTV safety monitor. A recorded video is treated
as a livestream: cut into 5 s chunks (1 s overlap, 4 fps), each chunk captioned + severity-scored
by a VLM (HERMES = Qwen3-VL-4B with a hierarchical streaming KV cache), with a PySide6 UI for
visual inspection and ground-truth matching.

The **JEV gate** sits in front of the VLM: a small hazard model (OneJev-0.8B) scores 3 s windows
("ticks", 1.5 s stride) for five hazards; a pure-logic router decides per chunk whether the VLM
sees it at FULL frame rate or REDUCED (1 fps). Every chunk still reaches the VLM (fail-open:
missing/invalid JEV data routes FULL). The v3 integration ("Domain Calibration & Gate v2") makes
the gate's scores trustworthy and domain-calibrated:

| Phase | Idea | One-line implementation |
| --- | --- | --- |
| P0 | Fix the crashes blocking measurement | argparse flags + g1_report key names |
| A | Make scores trustworthy | fp32 LM-head upcast (port of qev `_upcast_head`) + scoring harness skeleton |
| B | Make JEV cheap enough for dense gating | shared-prefix tick evaluation (1 prefill + 5 text-only suffix branches) |
| C0 | Get domain ground truth | iSafetyBench loader + 67→5 hazard mapping worksheet |
| C | Calibrate thresholds on that data | `calibrate_report.py` (item-split scan, precision floors, adoption gates) |
| D | Make routing temporally robust | GateRouter v2 (streak / hysteresis / EWMA-log), default-inert |

Execution order rationale (design §4): A changes scores, so nothing scores before it; B makes
C's ~1,100-clip scoring run 3–5× cheaper; C calibrates against the final readout; D consumes C's
output. E (VLM-track experiments) is optional and was not started.

---

## 1. Phase P0 — crash fixes (no behaviour change)

**Idea.** The three-way benchmark could not run at all (F-23) and the G1 analysis script read
key names that the gate log never wrote (F-24). Fix both with zero behaviour change, so every
later phase has a working measurement path and a green baseline.

**F-23 (`compare_latency.py`).** The master passed `--watch-threshold`/`--full-threshold` that
argparse never defined (instant `AttributeError`), and the worker's `getattr(..., 0.78/0.82)`
fallbacks silently baked an uncalibrated operating point into the benchmark. Implementation:
- both flags defined (`float`, default `None`) in the single shared parser;
- master appends each to the `jev_hermes` worker command **only when set**;
- other engines get `parser.error` (jev_hermes-only flags);
- worker builds `GateConfig` **from set values only** — no flags means the library defaults
  (0.20/0.35), never the legacy 0.78/0.82 pair (those are D-30 legacy candidates, reported
  never trusted).

**F-24 (`vlm_test_ui/scripts/g1_report.py`).** The script partitioned log rows on `"kind"` and
read `media_path`/`decider_label`/`ts`; the GateLog writes `"type"`/`media`/`decider`/`time`.
Fixed the key names. **A third latent bug surfaced later** (Phase C): check-4 read `t.get("vec")`
on tick records, which only ever carry `probs` (§5.6 key list) — invisible while the partition
was broken (zero ticks collected). Fixed `vec`→`probs` with an inline comment; disclosed as a
deviation for developer ratification.

**Verification.** `--help` shows the flags; `--engine hermes --watch-threshold 0.5` →
`parser.error`; g1_report on a synthetic log now counts ticks/chunks (was zero). The pinned
R-2/R-3 baseline script asserts `w.gate_panel is None`, but v2 Phase 4 (UI wiring) was never
implemented so `MainWindow` has no such attribute — verbatim run passes the R-2 assertions then
AttributeErrors; adapted line `not hasattr(w, "gate_panel")` (strictly stronger) used for the
green baseline, disclosed for ratification.

---

## 2. Phase A — fp32 LM head (trustworthy scores)

**Idea.** `scratch_jev_worker_test.json` showed two different chunks with **bit-identical**
`jev_pmax` = 0.4073334336280823 (F-21). Root cause (from the official `qev` engine
`references/sources/qev/engine.py::_upcast_head`): the model runs bf16; slot logits at magnitude
25–50 are quantized to 0.125 steps, so two-option probabilities move in ~3 % steps and can tie
exactly. Fix: compute the LM head in fp32 — untie it from the embedding (clone), cast the
readout input via a forward pre-hook, leave the rest of the forward bf16.

**Implementation (`vlm_test_ui/app/core/jev_adapter.py`, `OneJevDecider.__init__`).** Inserted
immediately after `model.eval()`, code verbatim from design §5.1: `head.weight` (and bias)
cloned to fp32 `nn.Parameter(requires_grad=False)`,
`head.register_forward_pre_hook(lambda module, args: (args[0].to(torch.float32),) + tuple(args[1:]))`.
Fallback if the model has no separable head: `self.head_dtype = "model"` + a logged warning.
`self.head_dtype` is reported by the harness run_meta so every artifact records which readout
produced it. Added `import logging` (stdlib; the lazy-import rule I-5 concerns torch only).

**The harness (`isafety_harness.py`, repo root).** One clip = one calibration item (§5.3):
sample the whole clip with the *runtime tick numerics* (`sample_timestamps` / `read_frames_at` /
`resize_max_side` from `app.core.frame_sampler` — reused, not reimplemented), one decider call,
one labeled row per hazard; unmappable hazard-on-item pairs are omitted (`label=None` → row
skipped), never assumed negative. Unreadable clips are skipped **and counted**. CLI per §8 with
exit codes 0/2/3.

**Key implementation idea — logits capture without prompt duplication.** PROBE-1 needs raw
`(logit_a, logit_b)` per (item, hazard), but editing `OneJevDecider` to expose logits is outside
the change budget. Instead `_SlotLogitCapture` registers a **forward hook on the LM head module
itself**: the fp32 head is the last compute op before the runtime readout, so a hook sees exactly
the logits the slot softmax consumes. One decider call = five head passes (one per hazard
question, in `HAZARD_QUESTIONS` order), so capture order maps 1:1 to hazards. Zero duplication,
zero drift.

**Key implementation idea — prompt hash (D-28).** `compute_prompt_hash()` = SHA-256 over
`json.dumps(HAZARD_QUESTIONS, sort_keys=True)` + the state-block text + the system prompt
(glossary formula). The state/system texts are pinned copies of the decider's construction
documented as such; any drift on either side changes the hash and voids calibration. Observed
value starts `4cd9341857fdc94f…`. Echoed into the harness run_meta, and (via
`GateConfig.prompt_hash`, Phase D) into the GateLog `run_start.gate_config` and
`thresholds.json`.

Null-input and order-probe rows (§5.3) were implemented in Phase C's pass — see §5 below.

---

## 3. Phase B — shared-prefix tick evaluation (JEV 3–5× cheaper)

**Idea.** The v2 decider ran **five full forward passes per tick** — the video re-encoded each
time — making JEV cost 129–182 s per chunk (~55–100 % of the HERMES chunk cost). The qev engine
solves this: render the shared prefix (video + state block) once, prefill once, then run five
**text-only suffix branches** off deep-copied caches. Port of `qev/engine.py::_run_sequential`
and the `mm_engine.py` vision-position rule.

**Mechanics (`OneJevDecider._call_shared`, design §5.2):**
1. `MARKER = "@@JEV_PREFIX_MARK@@"`; prefix render = chat template of messages whose user text
   ends with the MARKER (`add_generation_prompt=False`); the five full renders use
   `add_generation_prompt=True`. `split_prompt_suffixes` cuts at the MARKER index and raises
   `ValueError("prompt prefix mismatch")` if any full render doesn't start with the cut — the
   video placeholder must stay inside the prefix (trap 10). A mismatch → tick error → fail-open
   FULL (v2 §10.2 semantics preserved).
2. One processor call + one prefill on the cut prefix (`logits_to_keep=1`, `use_cache=True`,
   `return_dict=True`, **no explicit position_ids** — the model computes them internally).
   `P_exp = input_ids.shape[1]` (the processor-expanded length).
3. **Position capture:** forward pre-hooks (`with_kwargs=True`) on every decoder layer record
   `kwargs["position_ids"]`; take the **last layer's** recorded tensor, `[:, -1]`. Rationale
   (mm_engine header): after the last vision block, text positions continue linearly on all
   three M-RoPE axes, so the suffix continues at last-position + 1 per axis —
   `suffix_position_ids(last, L)` returns `[3, 1, L]` with `last[a] + 1 + j`. Hook mechanics
   mirror `vlm/hermes_engine.py::_register_forward_hooks`. `_decoder_layers()` resolves the
   layer list via `model.language_model.layers` with fallbacks (real-model path verified only
   at DEV-B2).
4. Five suffix branches: `tokenizer.encode(suffix)`, `copy.deepcopy(prefix_cache)` per branch,
   forward with `attention_mask=ones(1, P_exp+len)`, captured positions, `logits_to_keep=1`;
   delete the copy after each branch. Slot readout/softmax unchanged (T=1.0 over the two slot
   logits; fp32 head from §5.1).
5. **`order` kwarg** (`*, order="ab"` keeps the `JevDecider` protocol intact): `"ba"` swaps the
   two option lines and reads slots as `[id_b, id_a]` — used only by the order-probe; runtime
   is always `"ab"`.
6. **Mode dispatch:** `self.mode` (default `"shared"`); `"fullpass"` keeps the v2 five-pass path
   (`_call_fullpass`) solely for the DEV-B2 equivalence measurement. Prompts are built in one
   place (`_build_messages`) so both paths are byte-identical for `"ab"` by construction.

**Testing without a model (the pattern that made this verifiable).**
`tests/test_jev_adapter_v2.py` bypasses `__init__` (`OneJevDecider.__new__`) and injects a
`_FakeModel`/`_FakeProcessor` whose fake layers **fire the registered hooks** with a known
position tensor — exercising the capture mechanism end-to-end on CPU. Device strings derive
from `inputs["input_ids"].device` so the shared path runs CPU-only under stubs and CUDA in
production. PREFIX-unit 1–4 + ORDER-1–2; fail-on-break proofs for four breaks (cut off-by-one,
mismatch check removed, positions without +1, order mapping ignored).

**The one thing small tests cannot prove** (per the developer's small-tests-locally policy):
per-tick probability equivalence on the real model — that is DEV-B2 (`max |p_suffix −
p_fivepass| ≤ 5e-3` on ≥ 20 items, both modes through `isafety_harness.py --decider-mode`),
queued for the server. If it fails, the fast path does not ship.

---

## 4. Phase C0 — iSafetyBench ground truth (schema, mapping, loader)

**Idea.** Calibrate thresholds on real domain data (D-24). The dataset download was deferred to
the server (storage), so the schema came from the **official sources** (developer-sanctioned):
`github.com/iSafetyBench/data` exposes everything needed.

**Discovered schema** (validated against F-26's counts — exact match):
- `annotations_hazard.json` / `annotations_normal.json`: JSON lists of
  `{video_name, caption, gt_actions[]}`; the file a clip appears in is itself signal
  (hazard-file vs routine-file).
- `actions.txt`: **98 routine actions in 8 groups + 67 hazard actions in 10 groups**.
- Videos hosted on HuggingFace (`raiyaanabdullah/isafety-bench`, cc-by-nc-sa-4.0); naming =
  `video_name` (e.g. `-0zRCAh9_N4_trim_0.mp4`), layout in the actual snapshot unverified until
  the server inspects it.

**Mapping worksheet (`isafety_mapping.json`, Appendix B schema) — DRAFT, developer approval
pending.** 15 of 67 hazard actions mapped:
`pedestrian_vehicle` ← 4 (person dragged/crushed by vehicle, collision, vehicle losing control);
`no_ppe` ← **EMPTY — the taxonomy has no PPE-violation action at all** (only routine
"putting on/taking off PPE"); `blocked_exit` ← "cluttered workspace"; `unsafe_load` ← 6
load-integrity actions; `fall_or_spill` ← 4 fall/slip/flood actions; 52 unmapped listed with
per-category justifications in `notes`. Consequence to flag at review: no_ppe will have **zero
positives** from this dataset.

**Loader (`isafety_loader.py`, repo root).** Pinned interface
`load_items(dataset_dir, mapping) -> list[CalibrationItem]` with
`CalibrationItem = {item_id, video_path, hazard_labels: {hazard: "pos"|"neg"|None}, tags, camera}`.
Design choices:
- **Loud validation** (honours D-24's "no code against a guessed format" in spirit): schema
  violations raise with file+record; any `gt_action` outside the mapping's three lists aborts
  ("silent partial mapping is forbidden", §9); mapping sections must be disjoint.
- Subset rule per §5.3: positive iff tags ∩ mapped[h]; clean-negative iff all tags routine
  (normal file); **strict exclusion** for hazard clips tagged with any unmapped action — and it
  additionally **reports how many excluded clips would be kept under the loose reading** so the
  developer can revisit the rule with real numbers.
- Videos resolved by one recursive filename index over `dataset_dir`; missing files are skipped
  and counted downstream (harness), zero videos found = loud abort.
- Smoke-tested on a synthetic dataset dir (subset rule, exclusion counting, unknown-tag abort).

---

## 5. Phase C — calibration report + plumbing

**Idea.** Selection is **recall-first subject to precision floors** (D-25: full floor 0.60,
watch floor 0.40, CLI-configurable), with hard adoption gates (holdout recall ≥ 0.95 AND
precision ≥ floor) so a weak operating point can never silently ship. The report is CPU-only,
stdlib-only, deterministic and seeded — an agent can always run it.

**`vlm_test_ui/scripts/calibrate_report.py`** implements §5.4 exactly: schema validation with
line numbers (exit 2); **split by item, never by row** (an item's five hazard rows stay
together — `pmax(item)` needs them), stratified by `hazardous`, seeded shuffle, odd stratum
extra item to tune; per-hazard diagnostics (AUC with ties 0.5, Brier and 15-bin ECE **copied
from `qev/calibrate.py`** definitions, 5-bin reliability, `weak_ranker` flag < 0.65); midpoint
candidate scan (+ min−ε / max+ε); smallest-candidate-meeting-floor selection with
`degenerate_watch` fallback; holdout report for the selected point **and both legacy points**
(0.20/0.35, 0.78/0.82) with per-hazard breakdown; `ADVISORY: legacy point dominates selected`
line; 1000× bootstrap CIs; null-input table (mean p_yes + logit offset `ln(p̄/(1−p̄))`,
report-only) and order-bias table; exit 3 + **no thresholds.json** on gate failure;
`--report-only` recomputes holdout metrics for an existing file (the quality anchor).

**Appendix D finding (reported, never edited):** the doc's inline precision labels for t=0.15
and t=0.25 are swapped relative to arithmetic (t=0.15 → 4/7, t=0.25 → 0.5); every operative pin
(AUC 0.625, candidate set, t=0.65 → 2/3, floor 0.60 selects 0.65) is self-consistent, and the
CAL tests pin those operative values.

**Harness completion (the compliance-review fix).** `--null-input` (one uniform mid-gray item,
12 frames @ 320×320, `item_id="null_0"`, five rows `label="null"`, `source="null_input"` — the
"(or one, if none)" branch since the loader surfaces no resolution buckets) and `--order both`
(a second `order="ba"` decider call per item; rows `source="order_probe"`, `order="ba"`) were
added and smoke-verified with a hook-firing stub. Implementation detail worth knowing: the
capture expectation doubles under `--order both` (10 logits), and `ba` rows carry
`source="order_probe"` — the first draft used the item's source and the smoke caught it.

**Plumbing.** `compare_latency.py --gate-config-json PATH`: master loads/validates via
`build_gate_config()` (fail-fast `parser.error`), passes the **path** to the `jev_hermes`
worker; worker builds `GateConfig(**json.load(...))`; explicit `--watch-threshold`/
`--full-threshold` override the JSON when provided (§5.6). `g1_report.py` check-4 reads its
thresholds from `run_start.gate_config` (hard-coded constants only as a warned fallback).
3WAY-2 tests cover the plumbing with a fake config file.

**Synthetic G1 rerun (DEV-C3 agent part).** `GatedBatchWorker` + `MockJevDecider` +
`RecordingAdapter` on `make_test_video`, records written through `gate_record_to_dict`, fed to
`g1_report.py --demo-rules`: **all six checks pass** under the current 0.20/0.35 defaults with
thresholds consumed from `run_start.gate_config` — 28.6 % frame savings, zero down-rated
violation chunks, early warning (4000 ms) precedes first above_full (8000 ms). Threshold-aware
per design: DEMO_RULES windows stay the pseudo-GT; expected tiers re-derive from the current
`GateConfig`.

---

## 6. Phase D — GateRouter v2 (robust routing, default-inert)

**Idea.** Three additive knobs make the router temporally robust without touching the v2
behaviour unless configured: **watch streak** (`route_min_consecutive_watch` — require N
consecutive watch-level chunks before lifting), **hysteresis exit** (`route_watch_exit_threshold`
— stay FULL while elevated and pmax ≥ exit, even below watch), **EWMA** (`ewma_alpha` —
log-only smoothing, never routed on). Plus `prompt_hash` flowing into the GateLog.

**Implementation.** `GateConfig` gains the four fields after `route_reduced_fps` with §5.5
validation (`0 < exit ≤ watch`; streak int ≥ 0, not bool; `0 < alpha ≤ 1`). Router state
`elevated` / `watch_streak` / `ewma`; the rule table (1, 2, 3-with-streak-gate, 3b, 4, 5, 6)
with the pinned per-rule state updates. EWMA updates only on decisions that carry a `vec`, is
attached to `RouteDecision.ewma`, and nothing routes on it. `RouteDecision` and `GateRecord`
gain trailing defaulted `ewma` fields; `gate_record_to_dict` appends `"ewma"` after
`"top_tick_id"` (the §6-documented extension — see deviations); `gated_batch_worker.py` received
**exactly its one permitted line** (`ewma=decision.ewma,` — verified by diff).

**Default-identity proof.** `tests/test_router_v2.py` embeds a **frozen copy of the v2
`decide`** as the oracle and asserts identity over **1000 seeded randomized tick sequences**
(mixed ok/error statuses) with all knobs at defaults, plus `ewma is None` everywhere. Semantics
discovery worth recording: with `route_min_consecutive_watch ≥ 2`, a watch-level plateau never
accumulates through clear chunks — rule 6 resets the streak; the streak builds through rules
4/5 ("+1 if watch_now"), so activation requires a *rise* into the watch band. That matches the
pinned table exactly and is documented in the tests.

**LOGX-1** pins the v3 dict key order (v2 order + trailing `ewma`), `GateConfig.to_dict()`
containing `prompt_hash`, and — via a real mock worker run — `GateRecord.ewma is None` at
default config.

---

## 7. Cross-cutting implementation notes

- **Lazy heavy imports (I-5/JEV-8):** torch/transformers only inside `OneJevDecider`;
  `suffix_position_ids` imports torch inside the function; the harness imports the loader
  lazily inside `load_items()` so the file imports cleanly before C0 landed. Audited by the
  suite's JEV-8 subprocess test after every phase.
- **Fail-open everywhere:** decider exceptions, invalid outputs, prompt-prefix mismatches and
  position-capture failures all surface as tick errors → chunk routed FULL `jev_incomplete` →
  run continues. JEV results never enter `session.events` (gate data lives in
  `gate_records`/`GateRecord` only).
- **No timeouts, ever (C-5/T-1):** the word list is scanned over new files and added diff lines
  before every report; shell `timeout` for one's own test runs is fine and never ships.
- **Device-derivation for testability:** GPU paths derive device from the inputs, so stubbed
  unit tests run CPU-only and production runs CUDA — aligned with the developer's requirement
  that CPU-verified code runs unchanged on GPU machines.
- **Reuse, don't reimplement:** the harness scores with the *same* `frame_sampler` numerics as
  the runtime tick path, so calibration scores and production scores are comparable by
  construction.

## 8. Known issues and environment findings (do not re-derive)

1. **Pre-existing flaky test** `tests/test_gated_worker.py::TestWin1BaselineEquivalence::
   test_win_1_frame_count_equivalence` — failed ~2/16 multi-file runs, 0/20 in isolation. Two
   independent decode passes compared frame-for-frame; a boundary seek-miss in one pass drops a
   frame (GATE-3's comment acknowledges boundary seek-misses). Untouched (additive policy);
   developer decision pending on whether to stabilise it.
2. **Pre-existing suite errors** — the 6 `test_hermes_engine.py` GPU tests error with
   bitsandbytes `Tensor.item() cannot be called on meta tensors` during the HERMES 4-bit load.
3. **The environment blocker, resolved to root cause:** the same crash reproduces in the freshly
   built clean container (torch 2.14.1+cu130, same transformers/accelerate/bitsandbytes trio) —
   it is the **pinned library trio itself**, not a dirty venv. Untested hypothesis: 6 GB VRAM
   forces the CPU-offload dispatch path that trips the bug, so a bigger GPU may not crash —
   test as-is on the server before bumping versions. Fixing pins is the developer's call
   (requirements/Dockerfile are theirs).
4. **Container suite artifact** — the gated-worker tests' 60,000-spin event budget is too small
   on the container's slower event loop (`no_outcome`); a direct diagnostic with a larger budget
   shows the worker completing with the identical routing summary. Not a code defect.
5. **`finished` arrives twice** in worker test runs (custom emit + native QThread teardown
   signal) — pre-existing pattern; tests tolerate it (`outcome[0]` / all-`finished` assertions).
