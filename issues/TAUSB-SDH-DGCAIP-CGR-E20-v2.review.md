# TAUSB-SDH-DGCAIP-CGR-E20-v2 Review Handoff

## Current workflow state

- Spec: approved on 2026-08-17.
- Branch: `codex/tausb-sdh-dgcaip-cgr-e20-v2`.
- Active CSV row: `GIT-SNAPSHOT-01`.
- Scientific state: implementation and local mechanical validation only; no D0,
  mechanism, victim training, or AP50 claim has been produced.

## Objective

Test whether clean-real-TAL-aligned non-target instance Bernoulli-JS divergence can
locate under-protected person-cooccurring instances, then use a fixed-budget
DG-CAIP protection route to reduce co-occurring non-target damage without erasing
the person shortcut.

## Frozen execution chain

1. Run read-only D0 from
   `ue_framework/configs/tausb_sdh_dgcaip_d0_v2.yaml`.
2. Pull and hash the D0 locator artifact.
3. Only if D0 passes, bind the mechanism template with
   `ue_framework.tools.bind_dgcaip_mechanism_config`.
4. Commit and independently review the D0-bound mechanism config.
5. Run matched `P1-R/P2-CAIP/P3-DIST/P4-DGCAIP` arms.
6. Only if P4 passes, bind and run fresh-victim sparse E20.

The mechanism template is intentionally non-runnable before a real D0 SHA256 is
bound. P1-R is additionally bound to the historical P1 state and metrics hashes;
the replay tolerance is frozen at absolute `1e-6` plus relative `1e-4`.

## Local evidence

- Broad SDH/DG-CAIP regression: `166 passed`.
- Python 3.8 AST plus in-memory compile: 9 implementation files passed.
- Runnable D0 config validation: passed.
- Mechanism template fail-closed and D0 binder tests: passed.
- Module entrypoints:
  - `python -m ue_framework.tools.run_tausb_sdh --help`: passed.
  - `python -m ue_framework.tools.bind_dgcaip_mechanism_config --help`: passed.
- Direct script-file invocation is intentionally not used; this repository's
  previously reviewed contract is the module entrypoint from `ue_project/`.

These checks are mechanical evidence only. They do not establish locator quality,
mechanism benefit, target-class unlearnability, or non-target AP50 preservation.

## Active risks and gates

- D0 may fail the pre-registered correlation, Q4/Q1, or coverage gate. If so,
  mechanism and E20 remain blocked and the diagnostic is retained.
- Warm-up component calibration may fail closed if a required protection component
  has no finite positive gradient on the frozen calibration batches.
- P1-R replay must match historical structure and numeric tolerances. Any mismatch
  blocks P4 state even if other mechanism metrics look favorable.
- The worktree contains substantial unrelated user changes and artifacts. Git
  operations must stage only the explicit DG-CAIP task paths.
- GPU experiments require a passing exact-snapshot pre-run review and capped
  shutdown controller.

## Resolved implementation issues

- Corrected the paper-to-project direction: high divergence receives more
  protection; low divergence is not upweighted.
- Used per-class sigmoid Bernoulli-JS rather than whole-image softmax KL.
- Fixed clean TAL instance aggregation and ignored invalid background-only GT
  indices while still failing closed for invalid foreground assignments.
- Made DG-CAIP temperature, structural tolerances, and ranking minimum explicit in
  the config-to-engine-to-loss parameter chain.
- Added historical P1 metrics hash and replay gate instead of treating a named
  feature-off arm as proof of no regression.
- Split D0 and mechanism into separate snapshots so the mechanism config can bind
  the real D0 artifact hash.

## Pre-run decision

Pending `PRERUN-REVIEW-01`; no remote run is authorized by implementation evidence
alone.

## Append-only execution log

- 2026-08-17: User approved the v2 Spec.
- 2026-08-17: Implemented Bernoulli-JS/KL, clean-TAL instance aggregation,
  DG-CAIP weighting, fixed-budget CGR, D0/four-arm runner, configs, binder, and
  tests.
- 2026-08-17: Local broad regression reached 166 passing tests; no GPU run started.
