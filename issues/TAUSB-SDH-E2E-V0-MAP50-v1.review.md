# TAUSB-SDH-E2E-V0-MAP50-v1 execution review

## Current workflow state

- Approved Spec: `docs/research/specs/TAUSB-SDH-E2E-V0-MAP50-v1.md`.
- Durable state source: `issues/TAUSB-SDH-E2E-V0-MAP50-v1.csv`.
- Active row: `FEASIBILITY-LOADER-01`.
- GPU state: not started by this workflow; local/no-GPU implementation and validation only.
- Branch: `codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3`.

## Objective scientific result

Obtain one real, paired, single-seed 20-epoch VOC AP50 result for the current complete SDH detector-aware method before any further carrier tuning. The result is directional feasibility evidence only, not a formal unlearnable-example claim.

## Active risks and blockers

- The r2 hiding checkpoint failed the original RMS-diversity and high-frequency scientific gates. Those failures must remain visible in every feasibility artifact.
- The formal `tausb_sdh` loader and 200-epoch protocol must remain fail closed.
- No remote mechanism, integration smoke, or E20 run may start before its dedicated pre-run review passes on a pushed commit.
- The worktree contains unrelated user changes; only explicitly scoped V0 files may be staged later.

## Pre-run decision

`pending`. No GPU command is authorized by the Spec/CSV alone.

## Final claim/evidence review

`pending`.

## Append-only execution log

- 2026-08-11: User approved the E2E V0 reset: collect real AP50 first and defer carrier optimization.
- 2026-08-11: CSV structure validated: canonical 28-column header, 23 rows, unique IDs, and final `REVIEW-01` row.
- 2026-08-11: Selected `FEASIBILITY-LOADER-01` as the first unmet dependency; implementation is restricted to an exact-protocol feasibility branch with formal rollback tests.
- 2026-08-11: Closed `FEASIBILITY-LOADER-01`. The loader requires the frozen r2 metrics/checkpoint hashes and exactly the two recorded failed checks; 16 focused tests, Python 3.8 AST parsing, and diff-check passed. Formal failed-gate loading remains rejected.
- 2026-08-11: Started `MECH-V0-STATE-01`; scope is limited to persisting the actual optimized P1 with truthful gate/provenance fields while leaving formal PASS-only persistence unchanged.
- 2026-08-11: Closed `MECH-V0-STATE-01`. The actual arm-tagged P1 is reloaded and checked for finite tensors and Linf before the feasibility payload is written. The payload preserves hiding FAIL and the actual mechanism decision; 51 focused SDH tests passed.
- 2026-08-11: Started `CONFIG-PAIRED-01` to bind the exact mechanism input and add isolated smoke/E20 C0/M1 runtime contracts without weakening the formal 200-epoch protocol.
- 2026-08-11: Closed `CONFIG-PAIRED-01`. A ready mechanism config and a post-mechanism binder now generate four exact C0/M1 configs. Smoke selection and labels are hash-bound; the V0 materializer verifies the state file plus seven provenance hashes. Formal E200 remains unchanged.
- 2026-08-11: Started `EVAL-COMPARE-01` to carry V0 provenance into metrics and compare two explicit full VOC20 metric files with frozen directional thresholds.
- 2026-08-11: Closed `EVAL-COMPARE-01`. Evaluation now records clean-val and paired-training hashes plus the full mechanism provenance chain. The comparator outputs all 20 AP50 values with C0-minus-M1 drop and preserves mechanism/hiding gate failures. Thirty-one focused tests passed.
- 2026-08-11: Started `LOCAL-VALIDATION-01`; no GPU work is permitted in this row.
- 2026-08-11: Closed `LOCAL-VALIDATION-01`. A 103-test SDH regression suite passed, as did scoped compile, Python 3.8 AST, CLI, CSV, and diff checks. Read-only real VOC selection produced 200 records with 40 person and 160 person-free (`ced5d8ce...bcc7`).
- 2026-08-11: Started `GIT-SNAPSHOT-01`; only the V0 Spec/CSV/review, implementation, configs, tools, and four test files are in scope.
