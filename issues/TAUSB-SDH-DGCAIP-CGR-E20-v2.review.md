# TAUSB-SDH-DGCAIP-CGR-E20-v2 Review Handoff

## Current workflow state

- Spec: approved on 2026-08-17.
- Branch: `codex/tausb-sdh-dgcaip-cgr-e20-v2`.
- Active CSV row: `GIT-SNAPSHOT-D0-METRIC-01`.
- Last pushed snapshot: `503e09fbe707feef50206d9e53cdde9e553453a6`.
- Local worktree contains the reviewed raw-damage metric correction and
  data-disk cache hardening; a new exact commit is pending.
- Remote state: no D0 or GPU run has started.
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

- DG-CAIP focused regression after the raw-damage correction: `41 passed`.
- Broad SDH/DG-CAIP/NLA/CGR regression after the correction: `170 passed`.
- Python AST/in-memory compile for the six changed Python files: passed.
- D0 config contract (`VOC20`, target id `14`, `run_mode=d0`, no EOT): passed.
- Both D0 controller scripts passed Git Bash `bash -n`.
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

### PRERUN-REVIEW-01

- Result: blocked
- Decision: do_not_run
- Gated run: `REMOTE-D0-01`
- Code snapshot: `b682f9d428a426d0cca62f5a6a0fc599a0853b4e`
- Blocker: the implementation had an internal 1200-second guard but no independent
  outer timeout, tmux handoff, terminal evidence, or all-terminal shutdown wrapper.
- Resolution: added a data-disk-only D0 controller and launch gate; a new exact
  snapshot and `PRERUN-REVIEW-D0-02` are required before launch.

### PRERUN-REVIEW-D0-02

- Result: blocked
- Decision: do_not_run
- Gated run: `REMOTE-D0-01`
- Code snapshot: `503e09fbe707feef50206d9e53cdde9e553453a6`
- Blocker: the D0 locator and held-out Q1-Q4 summaries consumed the
  post-tolerance hinge losses (`0.005 / 0.02 / 0.05`) instead of the approved
  raw positive probability, IoU, and relative TAL-alignment damage. This could
  create artificial zeros and invalidate Spearman and quartile gates.
- Resolution implemented locally: raw positive damage is now carried separately
  for diagnostics, while optimization and nonlinear backtracking retain the
  tolerance-aware hinge losses. A regression test keeps hinge losses at zero
  while raw damage varies and verifies that the locator still ranks the damage.
- Additional hardening: CUDA, Matplotlib, Hugging Face, Torch, YOLO, XDG, and
  temporary caches are all routed to the AutoDL data disk; terminal evidence no
  longer overwrites an existing wrapper record.
- GPU job started: false
- Required next gate: new exact commit followed by `PRERUN-REVIEW-D0-03`.

## Append-only execution log

- 2026-08-17: User approved the v2 Spec.
- 2026-08-17: Implemented Bernoulli-JS/KL, clean-TAL instance aggregation,
  DG-CAIP weighting, fixed-budget CGR, D0/four-arm runner, configs, binder, and
  tests.
- 2026-08-17: Local broad regression reached 166 passing tests; no GPU run started.
- 2026-08-17: Created the scoped 21-file local commit `b682f9d`; ordinary push
  was not attempted again after the external-write authorization gate rejected it.
- 2026-08-17: User explicitly authorized the concrete destination; remote SHA was
  verified as `b682f9d428a426d0cca62f5a6a0fc599a0853b4e`.
- 2026-08-17: First pre-run review blocked D0 because an independent outer cost and
  shutdown controller was missing; implemented the bounded two-script fix locally.
- 2026-08-17: Pushed and verified the controller snapshot
  `503e09fbe707feef50206d9e53cdde9e553453a6` by ordinary non-force push.
- 2026-08-21: Independent D0 review found the raw-damage/hinge metric mismatch;
  no remote run was started.
- 2026-08-21: Implemented the metric separation and cache-path hardening locally;
  focused tests passed 41/41 and the broad related regression passed 170/170.
