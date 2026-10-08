# Start prompt — paste this into a fresh agent session to run the v3 integration

```text
Use the jev-integration skill.

Project: C:\Users\Maha Mamdouh\Desktop\New folder (3)\New folder\UI_TEST
Python: F:/URCA_PROJECTS/.venv/Scripts/python.exe

Read the skill's SKILL.md, then references/design_v3.md and references/verification_v3.md,
in full, before doing anything else. (references/design.md is the v2 historical contract —
consult it for baseline facts and traps, but design_v3.md governs.)

Start at Phase P0 of design_v3.md §11 and run the working loop of verification_v3.md §2
exactly: environment record, regression suite BEFORE, then the P0 budget (tree normalisation
stop-question to me first, then the compare_latency argparse fix and the g1_report fix),
regression suite AFTER, ledger row, phase report — and STOP for my explicit go (D-15).

Hard rules you must not break:
- Change budget design_v3 §7 is exhaustive; anything else is a stop-and-ask.
- GPU/model/network jobs: never run one unless I explicitly authorise that exact command;
  otherwise leave the prepared command behind and report NOT VERIFIED until I run and clear
  it (D-32).
- The regression suite is additive and must be green before and after every phase (D-27).
- No timeouts in product code (T-1 scan before every report).
- Anything the design is silent on is a question for me, not a decision.
```

Notes:
- The skill is installed at user scope (`~/.agents/skills/jev-integration/`); after editing this
  package copy, sync the installed copy so fresh sessions pick up v3:
  `cp jev-integration/SKILL.md ~/.agents/skills/jev-integration/SKILL.md` (and the
  `references/` files the same way), or reimport the package zip.
- Later sessions: same prompt, but replace the "Start at Phase P0" paragraph with e.g.
  "Phase A is approved; continue with Phase A" — the skill's "Where to start" rule makes the
  agent verify from the repo state and the metrics ledger that earlier phases are actually
  complete and their DEV items cleared.
- v2 history: references/design.md + verification.md remain for the v2 phases already shipped.
