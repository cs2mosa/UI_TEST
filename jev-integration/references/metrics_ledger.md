# Metrics ledger — JEV v3 (append-only; one row per phase; `pending` until the DEV item clears; never rewrite history)

| Phase | Date | M1 recall | M2 precision | M3 saved % | M4 jev s/tick | M5 early-warn | M6 HERMES | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | 2026-10-07 | n/a | n/a | n/a | pending (DEV baseline anchor handed off; reference band ≈ 25–36 s/tick from `scratch_jev_worker_test.json`: 129.4 s and 182.0 s per chunk across ~5–6 ticks) | n/a | n/a | P0 report |
| A | 2026-10-07 | n/a | n/a | n/a | pending (DEV-A1 PROBE-1 handed off; execution requires the C0 mapping — see A report §Deviations) | n/a | n/a | A report |
| B | 2026-10-07 | n/a | n/a | n/a | pending (DEV-B2 equivalence + latency anchor handed off; gate: ≥2× improvement vs the P0 baseline anchor) | n/a | n/a | B report |
| C0 | 2026-10-07 | n/a | n/a | n/a | n/a | n/a | n/a | C0 report (mapping draft + loader delivered; DEV-C1 download deferred to server by developer) |
| C | 2026-10-07 | pending (DEV-C2 scoring on server) | pending (DEV-C2 scoring on server) | n/a | n/a | n/a | n/a | C report (calibrate_report.py + CAL green; --gate-config-json plumbing; G1 rerun 6/6 OK) |
| D | 2026-10-07 | n/a | n/a | n/a | n/a | n/a | n/a | D report (router v2; ROUTE2/LOGX green ×2; frozen-v2 identity 1000 seqs; suite 176 green ×2; LOG-4 ewma-key extension disclosed) |
