# TAUSB-SDH-DGCAIP-CGR-E20-v2 Review Handoff

## Current workflow state

- Spec: approved on 2026-08-17.
- Branch: `codex/tausb-sdh-dgcaip-cgr-e20-v2`.
- Active CSV row: `GIT-SNAPSHOT-MECHANISM-01` (local scoped snapshot pending).
- Reviewed and pushed snapshot:
  `81bf37e5b19b318ffbfee18edbbf2071e69702dc`.
- GitHub remote branch SHA was independently verified as the same full commit.
- Remote state: D0 completed successfully, automatically requested shutdown, and
  its five-file minimal evidence set has been pulled and hash-verified.
- Scientific state: the diagnostic-only D0 locator gate passed. No mechanism-arm,
  victim-training, or AP50 improvement claim has been produced.

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

## Remote no-card preflight (2026-08-21)

- AutoDL data disk `/root/autodl-tmp` is mounted.
- Runtime imports passed with Python 3.8.10, PyTorch 2.0.0+cu118,
  Ultralytics 8.4.33, and PyYAML 6.0.
- `torch.cuda.is_available()` is false, as expected in no-card mode; no D0 or
  training command was run.
- Surrogate, historical P1 state, and hiding checkpoint SHA256 values match the
  frozen config.
- VOC train input contains 16,551 images and 16,551 labels.
- Free space: 17 GB on the data disk and 21 GB on the system overlay.
- The source worktree is intentionally not used as the execution checkout. Its
  current old branch lacks the secret manifest, while the reviewed commit tree
  contains the manifest and assets; the launcher will create a clean detached
  checkout from the exact pushed commit.

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

### PRERUN-REVIEW-D0-03

- Result: pass
- Decision: allow_run
- Gated run: `REMOTE-D0-01`
- Code snapshot: `81bf37e5b19b318ffbfee18edbbf2071e69702dc`
- GitHub branch binding: verified
- Local validation: 41 focused and 170 broad related tests passed
- Remote exact-checkout validation: passed with Python 3.8 config/CLI imports,
  raw-damage/zero-hinge locator probe, finite poison-only backward, input hashes,
  split hash, Bash syntax, and non-overwrite checks.
- Remote `pytest` is not installed; the local 170-test regression was therefore
  not duplicated remotely. This is an environment gap, not a failed test.
- Exact review packet:
  `research_workspace/experiments/TAUSB-SDH-DGCAIP-S0-E20/pre_run/prerun_d0_review_v3.md`.
- Blockers: none once the user explicitly enables GPU mode.

## D0 result and mechanism binding (2026-08-21)

- Controller status: `completed`, exit code `0`, exact execution commit
  `81bf37e5b19b318ffbfee18edbbf2071e69702dc`.
- Wrapper status: exit code `0`, automatic shutdown requested.
- Fatal scan: no Traceback, CUDA OOM, NaN/Inf, fatal, or error match.
- Eligible/covered non-target instances: `97/97`; finite coverage `1.0000`
  (gate `>=0.95`).
- Spearman between divergence and composite damage: `0.7295524`
  (gate `>=0.35`).
- Q4/Q1 composite-damage ratio: `4.0649343` (gate `>=1.5`).
- D0 decision: `pass`. This supports locator quality only and is not AP50 or
  fresh-victim evidence.
- Canonical pulled locator SHA256:
  `911896f16514639b3b5190d86155f6a48c7711ec1e206828b8e98674114d7539`.
- Five required files were pulled; remote/local hashes matched 5/5 and
  `missing_required=[]`.
- Bound mechanism config:
  `ue_project/ue_framework/configs/tausb_sdh_dgcaip_mechanism_v2.yaml`.
- Bound config SHA256:
  `aef8b3c04b58066adb79dcf239e2b9f124f6c50c3b7d8e3b324d9c25bb322a3e`.
- Binding retained the frozen split, P1 state, and P1 metrics hashes; no loss
  weight or threshold was tuned from D0. Focused binding/DG-CAIP tests: 28/28.

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
- 2026-08-21: Created local exact snapshot
  `81bf37e5b19b318ffbfee18edbbf2071e69702dc` with nine scoped files; push is
  pending explicit authorization for this new SHA.
- 2026-08-21: Completed read-only AutoDL no-card input/runtime/disk preflight;
  no artifact directory or remote run was created.
- 2026-08-21: User authorized and Codex completed an ordinary non-force push of
  `81bf37e5b19b318ffbfee18edbbf2071e69702dc`; GitHub remote SHA matched exactly.
- 2026-08-21: Two post-push SSH attempts returned `Connection refused`;
  `PRERUN-REVIEW-D0-03` remained in progress pending an online no-card instance.
- 2026-08-21: After no-card restart, fetched the exact commit into an isolated
  detached checkout and completed Python 3.8 manual sink, full runtime-input,
  split/hash, Bash, path-collision, and storage checks. Pre-run review v3 passed;
  no D0 or GPU process was started.
- 2026-08-21: User enabled GPU mode. The GPU prelaunch gate passed on an RTX
  4090D with free formal paths, and the exact `81bf37e5` launcher handed D0 off
  to tmux session `tausb-dgcaip-d0-s0-r1`.
- 2026-08-21: The first and second post-launch SSH checks both returned
  `Connection refused`, consistent with the controller's automatic terminal
  shutdown. Completion versus fast failure is not yet claimed; do not restart.
  Re-enable no-card mode and pull controller status, outer log, terminal record,
  and any D0 locator artifact first.
- 2026-08-21: After no-card restart, terminal evidence proved D0 completed with
  controller/wrapper exit code 0 and requested automatic shutdown. The fatal
  scan was empty.
- 2026-08-21: Pulled five exact D0 evidence files; all remote/local SHA256 values
  matched and no required item was missing. D0 passed all three preregistered
  locator gates (Spearman 0.7296, Q4/Q1 4.0649, coverage 100%).
- 2026-08-21: Bound the canonical D0 locator hash and remote path into the exact
  mechanism config without metric tuning; 28 focused tests passed. A new scoped
  local Git snapshot is next and requires separate authorization before push.
