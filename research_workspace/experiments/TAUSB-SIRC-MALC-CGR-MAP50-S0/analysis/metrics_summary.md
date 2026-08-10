# TAUSB-SIRC-MALC-CGR-MAP50-S0 metrics ingest boundary

The `exp-results-ingest-local` precondition is not met: the approved mechanism gate failed before C0/M1, so no victim `metrics/metrics.json` exists. The ingest script was not run and no AP50 value was synthesized from logs or mechanism diagnostics.

## Available mechanism metrics

- A0/A1 held-out mechanism summaries: verified and analyzed in `mechanism_gate.md`.
- Mechanism gate: FAIL.
- Frozen A1 carrier: absent.
- `allow_fresh_victim`: false.

## Expected validation gaps

- C0 clean-victim AP50: not run because of the pre-registered gate.
- M1 victim AP50 and all 19 non-target per-class values: not run because of the gate.
- `poisoned_count`, PSNR, LPIPS and materialized Linf: unavailable because M1 materialization did not run.
- Multi-seed evidence: outside this single-seed mechanism gate.

These are controlled stop-condition outcomes, not missing files from a completed victim experiment.
