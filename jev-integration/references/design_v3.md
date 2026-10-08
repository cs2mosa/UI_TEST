# JEV Domain Calibration & Gate v2 — Design Document

**Version:** 3.1 · **Date:** 6 Oct 2026 · **Repo baseline:** `UI_TEST` @ `722cbc7` ("updating the prompts"). Expected dirty state at session start: modified `calibrate_jev.py`, modified `compare_latency.py`, untracked `jev-integration/`. Phase P0 normalises the tree before any other work.
**Supersedes:** v2 (`design.md`) for everything covered here. v2 remains the historical contract for its completed Phases 1–3/6. v2's Phase 4 (GatePanel + `main.py`/`main_window.py` wiring) and Phase 5 (DEV UI runs) stay **open v2 work, not re-scoped here**; the v3 DEV vehicle is the CLI/worker path, never the UI.
**Status:** Developer decisions 6 Oct 2026 — register D-24..D-33. Implementation proceeds phase-by-phase (§11), stopping after every phase.

---

## 0. How to read this document

Tag discipline as v2 §0.1: **[GIVEN]** (developer, binding) · **[VERIFIED]** (observed in code/data) · **[PROPOSED]** (binding after register confirmation) · **[OPEN]** (ask; never pick yourself). The v2 §0.2 agent contract carries over verbatim: change budget exhaustive, never assume, stop and ask, tests before code, no features beyond this document. Every pinned name, signature, default, format string and expected test value in this document is **unchangeable**; if one looks wrong, report — do not edit.

**Prime directive (D-27):** no regression — detection quality *or* latency may only hold or improve, proven at every phase boundary by the regression protocol (§10.2). The regression suite is **additive**: a test that enters `vlm_test_ui/tests/` is never deleted, weakened, or skipped; a failing suite is a stop condition.

**GPU/network/model rule (D-32):** the agent never launches GPU, model-loading, or network jobs. Every such job is a **DEV item**: the agent prepares the exact command and an expected-output checklist and reports the item **NOT VERIFIED**. The item becomes VERIFIED only when the developer runs it and clears the results, or explicitly pre-authorises that specific run in a message (then the agent runs exactly that command, pastes the output, and marks it `VERIFIED (agent-run, developer-authorised)`). The agent never upgrades a NOT VERIFIED item on its own judgement.

---

## 1. Purpose and scope

### 1.1 Goal
Make the JEV gate **decide actively on our domain** (industrial/warehouse safety violations): repair the score-readout defects that make current scores untrustworthy, remove the JEV latency overhead that blocks dense gating, calibrate the routing thresholds against labeled domain data (iSafetyBench), make the router temporally robust — all training-free, in-process, regression-gated, phase by phase, each phase leaving the system measurably better. **[GIVEN]** (D-24..D-27)

### 1.2 Constraints
| ID | Constraint | Tag |
| --- | --- | --- |
| C-5, C-11, I-1..I-6 | Carried from v2 unchanged: no timeouts; in-process OneJev via transformers, CUDA-only; `--jev off` path byte-identical; lazy torch import; additions-only culture. | **[GIVEN]** |
| C-12 | **Training-free:** no weight updates, no gradients, no RL. Post-hoc calibration (thresholds, offsets, temperature), prompt/state engineering, and pure-logic router changes are allowed. Activation steering is **out of scope** (developer: "lets leave it"). | **[GIVEN]** |
| C-13 | **No regression** (D-27), enforced by §10.2. | **[GIVEN]** |
| C-14 | Calibration data: **iSafetyBench** (Abdullah et al., ICCVW 2025; ~1,100 industrial clips, multi-label tags, 98 routine + 67 hazard categories). | **[GIVEN]** (D-24) |
| C-15 | Operating point: **recall-first subject to a precision floor** (D-25). | **[GIVEN]** |
| C-16 | Phase E is **optional** and starts only after A–D are verified and running smoothly. | **[GIVEN]** (D-26) |
| C-17 | GPU/network jobs: permission protocol of §0 (D-32). | **[GIVEN]** |

### 1.3 Headline metrics (gates in Appendix A)
| Metric | Meaning | Must |
| --- | --- | --- |
| M1 `recall_full` | Fraction of hazard-positive items routed FULL | ≥ 0.95 on iSafetyBench holdout |
| M2 `precision_full` | Fraction of FULL-routed items that are hazard-positive | ≥ precision floor |
| M3 `frames_saved_pct` | Frame economy vs full-rate baseline | ≥ legacy point on same footage |
| M4 `jev_time_s`/tick | JEV cost per tick | ↓ ≥ 2× in Phase B; ±10 % noise band elsewhere |
| M5 `early_warning` | first above_watch/rising precedes first above_full | preserved (G1 check 6) |
| M6 HERMES quality/latency | GT match rate, `vlm_time_s` | Phase E items only: no metric worse, ≥ 1 better |

### 1.4 Non-goals (an agent MUST NOT implement)
- Any gradient-based fitting; activation steering (C-12); HTTP/ports (C-11); timeouts (C-5).
- v2 Phase 4 UI wiring; new UI widgets; changes to `vlm/**` or HERMES engine internals.
- DeepStream integration; Stage 2 live-stream simulation.
- Replacing `MockJevDecider` in unit tests (v2 D-21 stands); applying the null-input offset or EWMA to routing (log/report-only this cycle).

---

## 2. Baseline facts (current tree, 6 Oct 2026)

Method: **R** = read the file; **D** = read the data file; **W** = web (project page / arXiv 2508.00399).

| ID | Fact | Method |
| --- | --- | --- |
| F-20 | `OneJevDecider` (app/core/jev_adapter.py) loads bf16 and reads `out.logits[0, -1][[id_a, id_b]].float()` — **no fp32 head upcast**. The official `qev` engine (`references/sources/qev/engine.py::_upcast_head`) documents that bf16 logits at magnitude 25–50 quantize to 0.125 steps, moving two-option probabilities in ~3 % steps and letting them tie exactly. | R |
| F-21 | `scratch_jev_worker_test.json`: chunks 0 and 1 report **bit-identical `jev_pmax` = 0.4073334336280823** on different frames — consistent with F-20's tie behaviour (Phase A probe confirms or refutes). Same file: `jev_time_s` = 129.4 s / 182.0 s per chunk (~25–36 s/tick) — JEV ≈ 55–100 % of the HERMES chunk cost. | D |
| F-22 | Router defaults: `route_full_threshold=0.35`, `route_watch_threshold=0.20`; the uncommitted `compare_latency.py` worker overrides to 0.78/0.82. Both pairs are **legacy candidates** (D-30) — never trusted, always reported for comparison. | R |
| F-23 | `compare_latency.py` master passes `--watch-threshold`/`--full-threshold` to the `jev_hermes` worker, but neither argument is defined in `argparse`: master raises `AttributeError`; the worker rejects unknown flags. The three-way benchmark cannot complete today. | R |
| F-24 | `vlm_test_ui/scripts/g1_report.py` partitions rows on `r.get("kind")` and reads `media_path`/`decider_label`/`ts`; the GateLog writes `"type"`/`media`/`decider`/`time`. As-is the report shows zero ticks/chunks. Check-4 thresholds are hard-coded constants instead of read from `run_start.gate_config`. | R |
| F-25 | `OneJevDecider.__call__` runs **five full forward passes per tick** (video re-encoded each time), always A=yes/B=no order, T=1.0, no order debias. The official engine prefills the shared state+video prefix once and runs text-only suffix branches (`engine.py::_run_sequential`, `mm_engine.py` header). | R |
| F-26 | iSafetyBench: ~1,100 real industrial clips; open-vocabulary **multi-label** tags (98 routine + 67 hazard categories, 18 top-level groups); single/multi-label MCQ; HuggingFace `raiyaanabdullah/isafety-bench`; code `github.com/iSafetyBench/data`; CC BY-SA 4.0 (site). Annotation-file format is **unknown until C0 inspects it** — nothing may be coded against a guessed format. | W |
| F-27 | v2 Phase 4 absent: no `gate_panel.py`, no `--jev` flags in `main.py`. v2 Phases 1–3/6 implemented and tested; `GatedBatchWorker`, gate modules, `session_model` additions all present. | R |
| F-28 | `GateRecord` is constructed field-by-field in `gated_batch_worker.py` (the only place). Adding an optional trailing field to `GateRecord` does not break the worker; persisting a new value through it requires the single listed edit of §7. | R |

---

## 3. Glossary additions

| Term | Definition |
| --- | --- |
| **calibration item** | One (frames, label) pair: an iSafetyBench clip or a null-input frame set, scored once by the decider — the tick analogue on independent clips. |
| **labeled-score row** | Appendix C schema; one hazard's `p_yes` for one item; `label ∈ pos/neg/null`; unmappable hazard-on-item pairs are **omitted**, never assumed negative. |
| **operating point** | The pair `(route_watch_threshold, route_full_threshold)`. |
| **precision floor** | Minimum item-level precision of "route FULL" for the selected full threshold (default 0.60; watch floor 0.40). |
| **legacy point** | `(0.20, 0.35)` (v2 defaults) and `(0.78, 0.82)` (uncommitted override) — reported in every scan, never adopted without the C gates. |
| **null input** | Uniform mid-gray frame set (12 frames, 320×320) measuring the context-free yes-prior (Zhao et al. 2021). |
| **prompt hash** | SHA-256 over `json.dumps(HAZARD_QUESTIONS, sort_keys=True)` + the state-block text + the system prompt, computed by the harness, echoed into `thresholds.json` and the GateLog (via `GateConfig.prompt_hash`). |
| **elevated / watch streak** | Router v2 state: previous decision was watch-level-or-above / count of consecutive decided chunks with `pmax ≥ watch`. |
| **DEV item** | A GPU/model/network job the agent cannot run (§0). Agent prepares command + expected output; developer runs and clears. |
| **metrics ledger** | `jev-integration/references/metrics_ledger.md` — one row per phase with M1–M6 and evidence pointers; append-only. |

---

## 4. Architecture and execution order

```
OFFLINE (developer machine, GPU):                              RUNTIME (app worker / benchmark worker):
  iSafetyBench (hf_cache, D-31)                                  GateConfig (legacy defaults until C adoption)
      │  isafety_harness.py (+ isafety_loader.py, C0)                │
      ▼                                                              ▼
  labeled_scores.jsonl ──► scripts/calibrate_report.py ────►  GateRouter v2 (streak, hysteresis, EWMA-log)
  (+ null-input rows,         AUC/ECE/Brier · midpoint scan        │  ticks via OneJevDecider v2:
   order-probe rows)          tune/holdout · legacy table          │   fp32 head (A) + shared prefix (B)
      │                        │                                   ▼
      └──► thresholds.json (prompt-hash pinned) ──► (--gate-config-json) GatedBatchWorker / jev_hermes worker
```

**Execution order: P0 → A → B → C0 → C → D → E(optional).** Rationale: A changes scores, so nothing scores before it; B makes C's ~1,100-clip scoring run 3–5× cheaper and is probability-equivalent; C therefore calibrates against the **final** readout; D consumes C's output; E last and optional.

**Whole-integration definition of done:** every phase DoD met (§11); metrics ledger shows no metric regressed and M1/M2/M3/M4 improved where gated; final full three-way benchmark (DEV) run from a scratch directory with the adopted point; final report with the complete NOT-VERIFIED→cleared table.

---

## 5. Normative logic

### 5.1 fp32 LM-head upcast (Phase A) — port of `qev/engine.py::_upcast_head`
In `OneJevDecider.__init__`, immediately after `model.eval()`:
1. `head = model.get_output_embeddings()`; if `head is None` or it has no `weight`: set `self.head_dtype = "model"`, continue (log a warning) — qev's fallback.
2. Otherwise:
```python
self.head_dtype = "float32"
head.weight = torch.nn.Parameter(head.weight.detach().clone().to(torch.float32), requires_grad=False)
if getattr(head, "bias", None) is not None:
    head.bias = torch.nn.Parameter(head.bias.detach().clone().to(torch.float32), requires_grad=False)
head.register_forward_pre_hook(lambda module, args: (args[0].to(torch.float32),) + tuple(args[1:]))
```
(The clone unties the head from the embedding; the rest of the forward stays bf16.) `self.head_dtype` is reported by the harness run_meta. Softmax stays T=1.0; `validate_probs` unchanged.

### 5.2 Shared-prefix tick evaluation (Phase B) — port of the `qev` sequential branch path
**Contract:** one processor call and one prefill (video + state) per tick; five text-only suffix branches off deep-copied caches; per-tick probabilities equal to the five-pass path within tolerance. v2 §10.1 (no timeout/retry/caching across ticks) and I-5 (lazy imports) stand.

New **pure** helpers in `jev_adapter.py` (testable without a model):
```python
MARKER = "@@JEV_PREFIX_MARK@@"

def split_prompt_suffixes(prefix_render: str, full_renders: list[str]) -> list[str]:
    """prefix_render is apply_chat_template output for messages whose user text ends with MARKER,
    rendered with add_generation_prompt=False. full_renders are the five per-hazard renders
    (add_generation_prompt=True). Every full render MUST start with prefix_render[:index_of_MARKER];
    otherwise raise ValueError('prompt prefix mismatch'). Returns the five string suffixes
    full_renders[i][len(prefix_cut):]."""

def suffix_position_ids(last_prefix_positions: "torch.Tensor", suffix_len: int) -> "torch.Tensor":
    """last_prefix_positions: [3] positions of the final prefix token (captured per §5.2 step 3).
    Returns [3, 1, suffix_len] where element [a, 0, j] = last_prefix_positions[a] + 1 + j."""
```
Per tick, `OneJevDecider.__call__(frames_rgb, fps, t_ms, *, order="ab")` (the keyword-only default keeps the `JevDecider` protocol intact):
1. Build the five per-hazard prompts exactly as today (same system prompt, same state block, same option block); render all five with `apply_chat_template(..., tokenize=False, add_generation_prompt=True)`; render the prefix (user text = state block + `MARKER`) with `add_generation_prompt=False`; `suffixes = split_prompt_suffixes(prefix_render, full_renders)`.
2. **Prefix forward:** call the processor **once** on the cut prefix render with `videos=[frames_pil]`, `video_metadata=[...]`, `videos_kwargs={"do_sample_frames": False, "cap_pixels_per_frame": True}`, `return_tensors="pt"`; move tensors to cuda; forward with `logits_to_keep=1`, `use_cache=True`, `return_dict=True` and **no** explicit `position_ids` (the model computes them internally). Keep `past_key_values`; `P_exp = input_ids.shape[1]`.
3. **Position capture:** register a forward pre-hook on each language-model decoder layer that records `kwargs["position_ids"]` (the same mechanism `hermes_engine.py` uses to override positions); run the prefix forward; take the recorded tensor of the **last** layer, `last_prefix_positions = recorded[:, -1]` (shape `[3]`); remove the hooks. (Rationale: after the last vision block, text positions continue linearly on all three axes — `mm_engine` header — so the suffix continues from the last prefix position + 1.)
4. **Suffix branches:** for each hazard h: `suffix_ids = tokenizer.encode(suffixes[h], add_special_tokens=False)`; `cache = copy.deepcopy(prefix_cache)`; forward `input_ids=suffix_ids`, `attention_mask=ones(1, P_exp + len(suffix_ids))`, `position_ids=suffix_position_ids(last_prefix_positions, len(suffix_ids))`, `past_key_values=cache`, `use_cache=True`, `return_dict=True`, `logits_to_keep=1`; read `out.logits[0, -1][[id_a, id_b]]` as today. Delete the copy after each branch.
5. **Slot readout and softmax:** unchanged (T=1.0 over the two slot logits; fp32 head from §5.1).
6. **`order` parameter:** `"ab"` renders as today; `"ba"` swaps the two option lines (A. no: … / B. yes: …) and maps slot ids accordingly (`p_yes = softmax over [id_b, id_a]` first element). Used only by the harness order-probe; runtime always `"ab"`.
7. **Equivalence gate (DEV-B2):** on ≥ 20 fixed frame sets spanning hazard-present and hazard-absent, `max |p_suffix − p_fivepass| ≤ 5e-3` per hazard (qev reports 3e-4 for its batch-vs-sequential in fp32; the bound absorbs bf16 kernel rounding). Above the bound: **stop and report** — the fast path does not ship. The five-pass comparison runs with the fp32 head in both paths.

### 5.3 iSafetyBench integration (C0 + C)
| Aspect | Normative decision |
| --- | --- |
| Download (D-31) | `snapshot_download('raiyaanabdullah/isafety-bench')` into `F:\URCA_PROJECTS\.hf_cache`; the printed snapshot path is recorded in the C0 report. **DEV item.** |
| C0 inspection | The agent reads the dataset's annotation files (structure, category ids, video-path layout, split files if any) and writes `isafety_loader.py` + `isafety_mapping.json`. **No code may be written against a guessed format** (F-26): C0 is read-only until the loader exists. |
| Mapping | `isafety_mapping.json` (Appendix B) maps each of the five hazards to iSafetyBench category ids; also lists `unmapped_hazard_categories` and `routine_categories`. **Developer approves the mapping before any scoring.** |
| Subset rule | Positive item (for hazard h): clip's tags intersect h's mapped categories. Clean negative (for all h): tags intersect **only** `routine_categories`. Excluded: clips tagged with any hazard category outside the five mapped sets (their unmapped rows would poison negatives). One clip may be positive for several hazards. |
| Scoring protocol | One clip = one item: `sample_timestamps(0, duration_ms, 4.0)` + `read_frames_at(..., rgb=True)` + `resize_max_side(·, 640)` (the runtime tick numerics, reused from `app.core.frame_sampler`); one decider call; emit one row per hazard with `label` per the subset rule; omit rows where the mapping gives no verdict. Unreadable clips are skipped and counted (never silently). |
| Extra rows | `--null-input`: one gray-frame item per distinct resolution bucket in the data (or one, if none), 12 frames @ 320×320, five rows with `label="null"`. `--order both`: a second decider call with `order="ba"` per item, rows tagged `order="ba"`, `source="order_probe"`. |
| Split | The **report** owns the split (deterministic, seeded, stratified §5.4 step 2); harness rows carry no split. |

### 5.4 Threshold selection — `scripts/calibrate_report.py` (CPU-only; stdlib only; no app imports)
Deterministic; seeded (`--seed`, default 42). Input: `labeled_scores.jsonl` (first line `"type":"run_meta"`, Appendix C).
1. **Load + validate:** every row conforms to Appendix C; violations abort with the line number. Build per-item maps: `pmax(item) = max(p_yes over the item's isafety rows)`; `hazardous(item) = any of its rows has label "pos"`.
2. **Split (by item, never by row):** strata = items stratified by `hazardous` (any-pos vs all-neg); within a stratum sort items by `item_id`, shuffle with `random.Random(seed)`, first half of the **items** → tune, second half → holdout (odd counts: extra item to tune). Every row of an item stays with its item — `pmax(item)` needs all of its hazard rows in one half. Null/order rows are excluded from the split.
3. **Diagnostics (both halves, reported separately):** per hazard — n_pos/n_neg, AUC (ties count 0.5), Brier, ECE (15 bins, qev.calibrate definition), 5-bin reliability table. Any hazard with tune AUC < 0.65 is flagged `weak_ranker` (its rows still enter the pooled scan).
4. **Routing scan (tune):** candidates = midpoints between adjacent distinct `pmax` values across tune items (plus `min−ε` and `max+ε`). For each candidate t: `routed = pmax ≥ t`; precision = TP/(TP+FP), recall = TP/(TP+FN) against `hazardous`. Precision is nondecreasing in t, recall nonincreasing.
5. **Selection (D-25):** `route_full_threshold` = smallest candidate with precision ≥ `--precision-floor` (default **0.60**). `route_watch_threshold` = smallest candidate with precision ≥ `--watch-precision-floor` (default **0.40**) and ≤ full threshold (if none, watch := full, flagged `degenerate_watch`). If no candidate meets the full floor: print `FAILED-TO-ADOPT (precision floor unreachable)` and exit code 3.
6. **Holdout report:** for the selected point **and both legacy points**, holdout precision/recall/F1 of the routing rule; per-hazard breakdown. **Adoption gate (hard):** selected holdout recall ≥ 0.95 AND precision ≥ floor. If a legacy point also meets both hard gates with a higher F1, the report prints `ADVISORY: legacy point dominates selected` — the developer decides. Gates failed → exit code 3, `thresholds.json` not written.
7. **Bootstrap:** 1,000 resamples of holdout items (seeded `random.Random(seed)`), percentile 95 % CI for holdout recall/precision and pooled AUC.
8. **Null-input table:** per camera (single "global" group when rows carry no camera): mean p_yes per hazard; proposed logit offset `o_h = ln(p̄/(1−p̄))` — **report only**; nothing consumes it this cycle.
9. **Order-bias table:** per hazard, mean |p_ab − p_ba| over order-probe pairs.
10. **Outputs:** printed report; `thresholds.json` (Appendix C) on success; `--report-only THRESHOLDS_JSON` recomputes holdout metrics for an existing file (the quality anchor of §10.2); `--out PATH` overrides the output name.

### 5.5 Router v2 (Phase D) — additive, default-inert
`GateConfig` new fields (after `route_reduced_fps`, all defaulted):
```python
route_watch_exit_threshold: float | None = None   # hysteresis exit; None = disabled
route_min_consecutive_watch: int = 0              # 0 or 1 = fire immediately (v2 behaviour)
ewma_alpha: float | None = None                   # None = EWMA off
prompt_hash: str = ""                             # Phase C; flows into GateLog via to_dict()
```
`__post_init__` additions: if `route_watch_exit_threshold is not None` then `0 < exit ≤ route_watch_threshold`; `route_min_consecutive_watch` int ≥ 0, not bool; if `ewma_alpha is not None` then `0 < alpha ≤ 1`.

`GateRouter` state additions: `elevated: bool = False`, `watch_streak: int = 0`, `ewma: dict[str, float] | None = None`. `decide(ticks)` — first matching rule wins:
| # | Condition | Tier/Reason | State updates after the decision |
| --- | --- | --- | --- |
| 1 | v2 rule 1 (empty / any non-ok tick) | FULL / `jev_incomplete` | `prev_vec=None`; `watch_streak=0`; `elevated` unchanged; hold unchanged (v2 semantics) |
| 2 | `ge(pmax, full)` | FULL / `above_full` | v2 hold set; `elevated=True`; `watch_streak=0`; `prev_vec=copy` |
| 3 | `watch_now := ge(pmax, watch)` **and** (`route_min_consecutive_watch <= 1` or `watch_streak + 1 >= route_min_consecutive_watch`) | FULL / `above_watch` | `elevated=True`; `watch_streak = watch_streak + 1`; `prev_vec=copy` |
| 3b | not rule 3, `route_watch_exit_threshold is not None`, `elevated`, `ge(pmax, exit)` | FULL / `above_watch` | `elevated` unchanged; `watch_streak=0`; `prev_vec=copy` |
| 4 | v2 rule 4 (rise ≥ `route_rise_delta` vs `prev_vec`) | FULL / `rising` | `elevated=True`; `watch_streak = watch_streak+1 if watch_now else 0`; `prev_vec=copy` |
| 5 | v2 rule 5 (`hold_remaining > 0`) | FULL / `hold` | hold decremented; `elevated` unchanged; `watch_streak = watch_streak+1 if watch_now else 0`; `prev_vec=copy` |
| 6 | otherwise | REDUCED / `clear` | `elevated=False`; `watch_streak=0`; `prev_vec=copy` |

**EWMA (log-only):** when `ewma_alpha` is not None and the decision carries a `vec`: `ewma = vec` if `self.ewma is None` else per-hazard `a*vec[h] + (1-a)*ewma[h]`; attached to `RouteDecision.ewma`. Nothing routes on it. **Default-identity:** with the three knobs at defaults, rule 3b never fires (exit None), rule 3 fires iff `watch_now` (≤ 1), streak/elevated influence nothing, `ewma` stays None — routing output is identical to the v2 router for every input sequence. The test oracle is a frozen copy of the v2 `decide` embedded in `test_router_v2.py`.

### 5.6 Threshold consumption and pinning
- `compare_latency.py --gate-config-json PATH`: master loads the JSON, validates it parses as `GateConfig(**data)`, passes the **path** to the `jev_hermes` worker; the worker builds `GateConfig(**json.load(...))`. Explicit `--watch-threshold`/`--full-threshold` flags (P0-defined) override the JSON when provided. Existing engines untouched.
- `GateConfig` defaults change only after C's adoption gate (D-30); until then the calibrated point lives in `thresholds.json`.
- The GateLog `run_start.gate_config` automatically carries `prompt_hash` once the field exists (zero worker edits); g1_report reads thresholds from there after C.

---

## 6. Data model

| Change | Detail |
| --- | --- |
| `RouteDecision` | trailing field `ewma: dict[str, float] \| None = None` (frozen dataclass stays frozen). |
| `GateRecord` | trailing field `ewma: dict[str, float] \| None = None`; **`gated_batch_worker.py` gains exactly one edit**: the `GateRecord(...)` constructor call adds `ewma=decision.ewma,` (D-29). |
| `gate_record_to_dict` | appends `"ewma"` after `"top_tick_id"` — the documented v3 extension of the v2 key order. |
| `GateConfig.to_dict()` | gains `"prompt_hash"` automatically (asdict) once the field exists. |
| `labeled_scores.jsonl` | first line `{"type": "run_meta", ...}` (Appendix C); then one row per (item, hazard). |
| `thresholds.json` | Appendix C schema. |
| `metrics_ledger.md` | append-only table (§3). |

---

## 7. Change budget (exhaustive; anything else is forbidden)

**New files**
| File | Content | Phase |
| --- | --- | --- |
| `isafety_loader.py` (repo root) | `load_items(dataset_dir, mapping) -> list[CalibrationItem]`; `CalibrationItem = {item_id, video_path, hazard_labels: dict[hazard, "pos"\|"neg"\|None], tags, camera}`; encapsulates the C0-discovered annotation format | C0 |
| `isafety_mapping.json` (repo root) | Appendix B worksheet; developer-approved | C0 |
| `isafety_harness.py` (repo root) | CLI per §8; scoring, null-input, order-probe, `--dump-logits`, `--decider-mode`; writes `labeled_scores.jsonl` + run_meta | A (probe mode), B (decider-mode flag), C |
| `vlm_test_ui/scripts/calibrate_report.py` | §5.4; stdlib only; no app imports | C |
| `vlm_test_ui/tests/test_router_v2.py` | ROUTE2-1..8 incl. frozen-v2 identity property | D |
| `vlm_test_ui/tests/test_jev_adapter_v2.py` | PREFIX-unit 1..4, ORDER-1..2 (mock-free pure tests) | B |
| `vlm_test_ui/tests/test_calibrate_report.py` | CAL-1..6 with Appendix D pinned values | C |

**Edited files (exact edits only)**
| File | Edit | Phase |
| --- | --- | --- |
| `app/core/jev_adapter.py` | §5.1 fp32 head; §5.2 helpers + shared-prefix `__call__` + `order` kwarg | A, B |
| `app/core/jev_gate.py` | §5.5 GateConfig fields + validation; router state + rules; EWMA | D |
| `app/core/types.py` | `RouteDecision.ewma`, `GateRecord.ewma` (trailing, defaulted) | D |
| `app/workers/gated_batch_worker.py` | **exactly one line**: `ewma=decision.ewma,` added to the `GateRecord(...)` call | D |
| `compare_latency.py` | P0: define `--watch-threshold`/`--full-threshold` (float, default None) in the parser; master passes them to the worker **only when set**; worker builds `GateConfig` from set values only (replacing the `getattr` fallbacks). C: `--gate-config-json PATH` (master + worker pass-through; explicit flags override). | P0, C |
| `vlm_test_ui/scripts/g1_report.py` | P0: `kind`→`type`; header keys `media`/`decider`/`time`. C: check-4 thresholds read from `run_start.gate_config` (fallback to the constants prints a warning). | P0, C |

Everything else — including `temper/**`, `vlm/**`, all widgets, `config.py`, `chunker.py`, `batch_worker.py` (except the single listed line in `gated_batch_worker.py`), `exporter.py`, existing tests — is untouched. The v2 T-1 no-timeout word scan applies to every new file and every added diff line.

---

## 8. CLI contracts

**`isafety_harness.py`** (DEV-only; agent never runs without D-32 authorisation)
```text
--dataset-dir PATH        (required) local iSafetyBench snapshot
--mapping PATH            (required) isafety_mapping.json
--model-path PATH         default: JEV_MODEL_PATH env, else the pinned OneJev snapshot
--out PATH                default "labeled_scores.jsonl"
--limit N                 score only the first N items (after the subset rule)
--null-input              add gray-frame items
--order both|ab           default ab; "both" adds order-probe rows
--dump-logits PATH        also write raw slot logits per (item, hazard)
--decider-mode shared|fullpass   default shared (the §5.2 path); fullpass = the v2 five-pass
                          path, kept only for the DEV-B2 equivalence measurement
--sample-fps 4.0  --max-side 640  --seed 42
```
Exit codes: 0 ok; 2 bad args/mapping; 3 no items survived the subset rule.

**`scripts/calibrate_report.py`** (agent-runnable; CPU, stdlib)
```text
labeled_scores.jsonl positional
--precision-floor 0.6   --watch-precision-floor 0.4   --seed 42
--out thresholds.json   (skipped on FAILED-TO-ADOPT)
--report-only THRESHOLDS_JSON   recompute holdout metrics for an existing file
```
Exit codes: 0 adopted; 3 gates failed; 2 schema/IO error.

**`compare_latency.py`** additions: `--watch-threshold FLOAT`, `--full-threshold FLOAT` (P0), `--gate-config-json PATH` (C). Master semantics for `jev_hermes` only; other engines reject both flags with `parser.error`.

---

## 9. Failure handling

| Situation | Behaviour |
| --- | --- |
| Prefix render / full render mismatch in `split_prompt_suffixes` | `ValueError("prompt prefix mismatch")` → tick error → chunk FULL `jev_incomplete`; run continues (fail-open, v2 §10.2). |
| Suffix equivalence above 5e-3 (DEV-B2) | Stop-and-ask; fast path not shipped. |
| Harness: clip undecodable / zero frames | Skip item, count in the summary line, continue. |
| Harness: mapping missing a category present in the data | Abort (exit 2) — silent partial mapping is forbidden. |
| Report: schema violation | Abort (exit 2) with row number. |
| Report: precision floor unreachable / adoption gates failed | Exit 3; message; no `thresholds.json`. |
| `--gate-config-json` file missing / invalid | `parser.error` at master; worker exits non-zero with the message. |
| Regression suite red at any point | Stop; fix or ask; never weaken a test (additive policy). |

---

## 10. Verification

### 10.1 Test catalogue (unittest style under `vlm_test_ui/tests/`; no GPU/network; no absolute paths; pure-logic tests need no QApplication)
| ID | Subject |
| --- | --- |
| R-1..R-3 | v2 baseline suite re-run after **every** phase (11 headless tests; R-2/R-3 script unchanged from v2 verification §2). |
| JEV-8 | After an R-3 run and a MOCK run, `torch`/`transformers` absent from `sys.modules` (I-5 still holds with the new `order` kwarg and helpers). |
| PROBE-1 (DEV) | fp32 head: on ≥ 20 fixed frame sets, no two distinct frame sets produce bit-identical (id_a, id_b) logits; logit-step histogram shows sub-0.125 resolution. Before/after distributions pasted in the A report. |
| PREFIX-unit 1..4 | `split_prompt_suffixes`: exact common-prefix cut; mismatch → ValueError; single-render passthrough; suffix text equals `full[len(prefix_cut):]`. `suffix_position_ids`: shape `[3,1,L]`; values `last+1+j` per axis. |
| ORDER-1..2 | With a stubbed forward (monkeypatched model/processor in `jev_adapter`-level unit test): `order="ba"` swaps slot ids so p_yes is invariant under the swap for a symmetric stub, and differs for an asymmetric one. |
| PREFIX-eq (DEV) | §5.2 step 7 equivalence ≤ 5e-3 on ≥ 20 fixed frame sets. |
| CAL-1..6 | Appendix D pinned cases: AUC with ties (0.625 example); midpoint candidates; precision-floor selection; stratified split determinism (same seed → identical split; odd-stratum extra to tune); legacy rows rendered; bootstrap reproducible; exit code 3 path. |
| ROUTE2-1..8 | Each new knob fires; each is inert at defaults; **frozen-v2 identity over ≥ 1,000 randomized tick sequences** (mixed ok/error statuses, seeded); EWMA arithmetic across chunks; streak reset on rule 1; hysteresis boundary uses `ge` EPS semantics; `hold_remaining_after` correctness with 3b present. |
| LOGX-1 | `gate_record_to_dict` key order = v2 order + trailing `ewma`; `GateConfig.to_dict()` contains `prompt_hash`; worker-emitted `GateRecord.ewma` is None at default config (mock run). |
| 3WAY-2 | `--gate-config-json` plumbing with a fake config file and injected fake runner/decider: worker builds the config, existing engine outputs untouched, explicit flags override JSON. |
| G1 rerun | Synthetic `DEMO_RULES` gated run (agent-runnable: `GatedBatchWorker` + `MockJevDecider` + `RecordingAdapter` on `make_test_video`, driven exactly as `test_gated_worker.py` does; the run's `gate_records` are written through `gate_record_to_dict` and fed to `scripts/g1_report.py`). **Threshold-aware:** checks 1–3 (histogram, frame economy, overhead) assert mechanically; checks 4–6 re-derive the expected tier of each window from the mock's probabilities **under the current `GateConfig`** (the DEMO_RULES windows stay the pseudo-GT labels; the fixed 0.35/0.20 assumptions do not survive recalibration). Every routing change vs the pre-adoption G1 is listed and explained. |
| T-1 | No-timeout word scan (v2 §10.1 list) over all new files and added diff lines. |

### 10.2 Regression protocol (run at every phase boundary; D-27)
**Agent part (every phase, before and after changes):**
```bash
export QT_QPA_PLATFORM=offscreen
cd "$REPO/vlm_test_ui"
"$PY" -m pytest tests/ -v -p no:cacheprovider        # full suite; known-hang test excluded as in v2 R-1
"$PY" -m pytest tests/test_gate_core.py tests/test_frame_sampler.py -v -p no:cacheprovider   # twice in a row
```
The suite must be green **before** the phase's first edit (snapshot) and **after** the last edit. Fail-on-break proof for each new test (v2 §12.2). Diff review + T-1 as v2 verification §4 (file list extended per §7).

**Developer part (DEV anchors; agent reports NOT VERIFIED until cleared):**
```bash
# latency anchor (3-chunk jev_hermes worker only; scratch directory!)
cd <scratch> && "$PY" "C:/Users/Maha Mamdouh/Desktop/New folder (3)/New folder/UI_TEST/compare_latency.py" \
    --mode worker --engine jev_hermes --video <CLIP> --num-chunks 3 --output anchor_<phase>.json
# quality anchor (after C):
"$PY" scripts/calibrate_report.py labeled_scores.jsonl --report-only thresholds.json
```
Anchors compare against the previous phase's cleared numbers; M4 tolerance ±10 % unless the phase gate says otherwise.

### 10.3 Evidence rule
v2 §12.3 verbatim: exact command, pasted output, R-1..R-3 re-run, `git diff --stat`, forbidden-file check, T-1 output, explicit NOT-VERIFIED list with handoff commands. A phase report also appends its row to `metrics_ledger.md`.

---

## 11. Implementation phases (stop after every phase — D-15)

| Phase | Content | Files | Definition of Done |
| --- | --- | --- | --- |
| **P0** | Tree normalisation + crash fixes, no behaviour change: (1) `git status --short` must show exactly the expected dirty set (§ header) — else stop; (2) **ask the developer** to commit the two modified files (recommended) or revert them; (3) implement the P0 column of §7 (argparse fix F-23, g1_report fixes F-24); (4) run the regression protocol (agent part) and hand the DEV latency anchor to the developer as the **baseline anchor**. | `compare_latency.py`, `scripts/g1_report.py` | Suite green before/after; anchor command handed off; report; ledger row 0 |
| **A** | §5.1 fp32 head; `isafety_harness.py` skeleton with `--dump-logits` probe; unit tests. Thresholds untouched (D-30). | `jev_adapter.py`, `isafety_harness.py` (new) | Suite green ×2; PROBE-1 handed off; probe results cleared → F-21 confirmed/refuted recorded in ledger |
| **B** | §5.2 shared-prefix + pure helpers + `order` kwarg + unit tests; PREFIX-eq handed off; latency report from cleared DEV run. | `jev_adapter.py`, `test_jev_adapter_v2.py` (new) | PREFIX-unit/ORDER green; PREFIX-eq cleared ≤ 5e-3; M4 ≥ 2× improvement recorded |
| **C0** | iSafetyBench download (DEV, D-31); read-only inspection; `isafety_loader.py` + `isafety_mapping.json`; **mapping review stop** (developer approves). | `isafety_loader.py`, `isafety_mapping.json` (new) | Mapping approved; subset counts reported |
| **C** | Harness scoring run (DEV) → `labeled_scores.jsonl`; `calibrate_report.py` + CAL tests; selection per §5.4; adoption decision; `--gate-config-json`; DEV-C3 domain-shift G1 + one real-footage gated run handed off. | `calibrate_report.py`, `test_calibrate_report.py` (new); `compare_latency.py` | CAL green; adoption gate §5.4(6) cleared; thresholds.json written; DEV-C3 cleared; ledger M1/M2/M3 updated |
| **D** | §5.5 router v2 + `ewma` fields + the single worker line; ROUTE2 tests incl. frozen-v2 identity; synthetic G1 identity at defaults. | `jev_gate.py`, `types.py`, `gated_batch_worker.py` (1 line), `test_router_v2.py` (new) | ROUTE2/LOGX green; G1 identical at defaults; regression anchors hold |
| **E (OPTIONAL — D-26)** | Only after A–D verified and smooth; each item separately gated by M6: (E1) disagreement-triggered verification re-ask in the `jev_hermes` worker; (E2) prior-caption continuity in the worker prompt; (E3) per-camera normality card; (E4) measure-only FULL-chunk resolution probe. Each item: implement → G0 table handed off → adopt only if no metric worse, ≥ 1 better. | `compare_latency.py` + prompt-assembly helpers | Per-item G0 cleared; ledger M6 rows |

DEV items inventory (all NOT VERIFIED until the developer clears): PROBE-1, PREFIX-eq, iSafetyBench download, harness scoring runs, DEV-C3, latency/quality anchors, Phase E G0 runs, final three-way benchmark.

---

## 12. Decision Register (v3)

| ID | Decision | Status |
| --- | --- | --- |
| D-24 | Calibration data source is **iSafetyBench**. Developer: "we will calibrate using the iSafetyBench dataset, it has annotations for short clips of violations that is relevant to our task, we can calibrate the thresholds on it and it could improve the metrics we use." | **Approved** 6 Oct 2026 |
| D-25 | Operating point = recall-first subject to a precision floor; floors default full **0.60**, watch **0.40** (configurable flags, no register churn needed to retune); hard adoption gates §5.4(6). Developer: "operating point should be what you recommended." | **Approved** (floor values **[PROPOSED]**, adjustable) |
| D-26 | Phase E optional; only after A–D verified and running smoothly. Developer: "phase E, put it in the doc and flag it as an optional phase whenever we test everything else and al can run smoothly." | **Approved** |
| D-27 | No-regression policy, enforced by §10.2; regression suite is **additive** (no test deleted/weakened/skipped). Developer: "make sure that no regression failures happen and the performance … doesnt get lower. only better"; "uses the regression tests and adds on it so we ever improve phase by phase". | **Approved** |
| D-28 | Prompt-hash pinning: thresholds valid only for the pinned prompt; hash in `GateConfig.to_dict()` → GateLog and in `thresholds.json`; any prompt edit voids thresholds → recalibrate. | **Approved** (blanket "okay for any updates") |
| D-29 | v2 I-3 amended: `jev_adapter.py`, `jev_gate.py`, `types.py`, `gated_batch_worker.py` (**one listed line only**), `compare_latency.py`, `scripts/g1_report.py` editable strictly within §7; all other v2-invariant files untouched. | **Approved** (blanket) |
| D-30 | Threshold re-derivation: v2 defaults and the uncommitted 0.78/0.82 are legacy candidates; `GateConfig` route defaults change only after C's adoption gate; Phase A voids prior score-based tuning. | **Approved** |
| D-31 | iSafetyBench download into `F:\URCA_PROJECTS\.hf_cache` = pre-approved network job (mirrors v2 D-19); executed by the developer or agent with explicit per-run authorisation (D-32). | **Approved** (implied by D-24) |
| D-32 | **GPU/network permission protocol** (§0): agent never launches GPU/model/network jobs; default is handoff (exact command + NOT VERIFIED); agent-run only on explicit per-run developer authorisation; results count only when the developer clears them. Developer: "when the agent hits a GPU run it should ask for permission from the pair developer or it leaves the test behind and reporting it as unverified until the developer runs it and clears the results." | **Approved** |
| D-33 | Activation steering is **out of scope** for v3 (developer: "lets leave it"); may be revisited only via a new register entry with its own gated experiment. | **Approved** |
| — | v2 register D-1..D-23 stands except where D-29 amends I-3. | **Carried** |

## Appendix A — Metric gates
| Metric | Source | Gate |
| --- | --- | --- |
| M1 recall_full | report holdout, pooled scan | ≥ 0.95 (selected point) |
| M2 precision_full | same | ≥ precision floor (default 0.60) |
| M3 frames_saved_pct | G1 check 2 / benchmark summary | ≥ legacy point on same footage |
| M4 jev_time_s/tick | worker telemetry | B: ≥ 2× ↓; others ±10 % |
| M5 early_warning | G1 check 6 | passes after adoption |
| M6 HERMES | G0 table / GT match | Phase E: none worse, ≥ 1 better |
| weak-ranker flag | report per-hazard AUC | < 0.65 flagged (report-only) |

## Appendix B — `isafety_mapping.json` schema
```json
{
  "version": 1,
  "hazards": {
    "pedestrian_vehicle": ["<category_id>", "..."],
    "no_ppe": ["..."], "blocked_exit": ["..."],
    "unsafe_load": ["..."], "fall_or_spill": ["..."]
  },
  "unmapped_hazard_categories": ["..."],
  "routine_categories": ["..."],
  "notes": "<per-category justification; reviewed by developer>"
}
```

## Appendix C — data schemas
`labeled_scores.jsonl` first line:
```json
{"type": "run_meta", "prompt_hash": "...", "model_path": "...", "head_dtype": "float32",
 "gate_config": {"...": "..."}, "dataset_dir": "...", "mapping_version": 1, "seed": 42}
```
Rows (one JSON object per line, `separators=(",", ":")`):
```json
{"source": "isafety|null_input|order_probe", "item_id": "clip_00123|null_0",
 "hazard": "no_ppe|fall_or_spill|...", "p_yes": 0.0, "label": "pos|neg|null",
 "tags": ["..."], "order": "ab|ba", "camera": null,
 "meta": {"duration_ms": 0, "n_frames": 0}}
```
`thresholds.json`:
```json
{"route_watch_threshold": 0.0, "route_full_threshold": 0.0,
 "precision_floor": 0.6, "watch_precision_floor": 0.4, "seed": 42,
 "prompt_hash": "...", "dataset_fingerprint": "<n_items/n_pos/n_neg per hazard>",
 "selection": {"tune_stats": {"...": "..."}},
 "holdout": {"selected": {"recall": 0.0, "precision": 0.0, "f1": 0.0},
              "legacy_020_035": {"...": "..."}, "legacy_078_082": {"...": "..."}},
 "null_input": {"global": {"no_ppe": 0.0, "...": 0.0}},
 "order_bias": {"no_ppe": 0.0, "...": 0.0},
 "created": "<ISO-8601>"}
```

## Appendix D — pinned CAL test values
Items (single hazard `no_ppe`; item-level pmax shown): pos `[0.9, 0.8, 0.3, 0.2]`, neg `[0.7, 0.6, 0.4, 0.1]` — eight items, one row each.
- AUC = (0.9 beats 4 negs) + (0.8 beats 4) + (0.3 beats 1) + (0.2 beats 1) = 10/16 = **0.625**.
- Candidates (midpoints of adjacent distinct pmax values): 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85.
- At t = 0.75: routed {0.9, 0.8} → precision **1.0**, recall **0.5**. At t = 0.65: routed {0.9, 0.8, 0.7} → precision **2/3**, recall **0.5**. At t = 0.25: routed all but 0.1 → TP 4, FP 3 → precision **4/7 ≈ 0.571**, recall **1.0**.
- With floor 0.60: candidates give precisions 0.5 (0.15), 4/7 (0.25), 0.4 (0.35), 0.5 (0.5), 2/3 (0.65), 1.0 (0.75), 1.0 (0.85) → smallest meeting the floor is **0.65** (recall 0.5). The toy set documents that floors can cost recall — the real scan runs on hundreds of items.
- Split determinism: same seed twice → identical item-level tune/holdout assignment; odd stratum size → extra **item** to tune; all rows of an item share its half.

## Appendix E — References
qev `engine.py::_upcast_head` / `_run_sequential` / `calibrate.py` (ECE/Brier definitions) · `mm_engine.py` header (position rule) · jev-calibrate (midpoint/precision workflow, holdout discipline) · Zhao et al. 2021 (contextual calibration / null input) · Holtzman et al. 2021 (DC-PMI) · iSafetyBench (arXiv 2508.00399, `isafetybench.github.io`, HF `raiyaanabdullah/isafety-bench`) · AnomalyRuler ECCV 2024 (E3) · LAVAD CVPR 2024 · v2 design.md (invariants, traps, T-1) · `scratch_jev_worker_test.json` (F-21).
