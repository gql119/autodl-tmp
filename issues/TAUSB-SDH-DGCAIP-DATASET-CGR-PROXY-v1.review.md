# TAUSB-SDH-DGCAIP-DATASET-CGR-PROXY-v1 review

## Scope

- Approved on 2026-08-31.
- Shortcut-causality validation is explicitly deferred.
- This review covers local implementation only. No GPU scan, mechanism, victim training, dataset materialization, or AP50 claim is authorized by a local PASS.
- Pre-existing dirty R4/P4 records and experiment artifacts are excluded from this change set.

## Initial review

Implementation is present for the three approved problems. Static review found
and corrected one material integration gap: e1/e5/e20 had originally affected
only the risk rank, while the strict protection rows still came from the main
surrogate. The strict path now keeps one independently named protection row per
snapshot and hashes all three frozen models before and after optimization.

The dependency-backed tests were executed on the authorized AutoDL no-card
instance from detached commit `f8fe99e5ef3ac19294ec1a932bd73c5ef2de63f3`.
The implementation review therefore passes; this does not authorize or imply a
GPU scientific result.

## Regression review

Code-path review confirms that legacy P4/R4 configs retain batch-local ranking
and their historical additive route functions. Only
`TAUSB-SDH-DGCAIP-DATASET-CGR-PROXY-v1` with `strict_mechanism` loads the frozen
dataset bank and selects strict final-update routing. The focused legacy
regression suite completed with exit code 0.

## Evidence

- `compileall`: PASS for all modified/new method and test modules.
- pure-Python dataset bank deterministic build, round trip, canonical hash, and
  tamper rejection: PASS.
- execution CSV shape: PASS, 8 rows and 28 columns. An accidentally concatenated
  row was corrected before execution.
- `git diff --check`: PASS apart from Windows line-ending notices.
- GPU: not started.
- new and modified DG-CAIP focused suite: PASS, 49 tests in 5.70 seconds.
- legacy sparse/P4, victim seed, P1 determinism, multi-parameter CGR,
  mechanism-objective and config regression: PASS, 73 tests collected, exit 0.
- remote review worktree: detached at `f8fe99e`, clean after tests; no GPU device
  was exposed in no-card mode.
- the G2 `short_victim_risk_scan` consumer and agreement gate are implemented;
  the actual 3-epoch fresh-victim job must be bound to the frozen G1 candidate
  and fixed train subset only after L0/G0/G1 pass.

## Decision

`pass_g0_binding_ready_for_c0_snapshot_refresh`

## G0 clean C0 snapshot binding

The existing clean C0 run cannot be used as the required three-snapshot
teacher without qualification. Its retained `epoch0.pt`, `epoch5.pt`, and
final checkpoint represent post-epoch 1, 6, and 20 states. The new gate
therefore refuses `epoch5.pt` as the e5 teacher and schedules one clean C0
refresh with `save_every_n_epochs=1`. The exact bindings after that run are:

- e1: checkpoint `epoch0.pt`, internal epoch index 0;
- e5: checkpoint `epoch4.pt`, internal epoch index 4;
- e20: checkpoint `epoch19.pt`, internal epoch index 19.

This refresh reuses the prior clean sparse input; it does not regenerate VOC,
run the DG-CAIP risk scan, generate poison, or train M1. The output is placed on
the AutoDL data disk. The controller has a 45-minute wall cap, a 15-minute
no-log-progress cap, fresh-initialization tensor-hash enforcement, checkpoint
metadata verification, and optional shutdown on every terminal exit.

## G0 no-card evidence

- execution commit: `77aefc7b5a3181c6881a8f96ac4adae17657aafd`;
- remote focused test: 13 passed in 3.72 seconds;
- config SHA-256: `e1c7992eb824d0a182bf69a2e42f795b1f82b1dde27b2db7c65d8fa8f551bfc0`;
- preflight evidence SHA-256: `2a4a150064ae180cb5ffff359ff08160d9bbad33cf8307a12f6777530bdea306`;
- clean sparse report SHA-256: `e017b0fc06aa899ae62ecdc4f03b3e5e7a3ef32d2024abfd26b572a096519fdf`;
- live audit: 16,551 total, 6,095 person, 0 poisoned;
- victim fresh-init expected tensor SHA-256:
  `54aaf431f8a67b3f3067319a8164d1d6db6874497a46109a17af76a15d2b994c`;
- five GPU-output roots were still absent after preflight;
- data disk free: 9,308,647,424 bytes;
- `nvidia-smi`: no device visible, so no GPU work was started.

The first preflight invocation omitted the `ue_project` working directory and
failed before importing the controller. The corrected invocation passed; no
experiment path was created by either invocation.

## Next gate

Enable GPU only for `C0-SNAPSHOT-REFRESH`. Run the reviewed controller at
commit `f5e223a` with `--shutdown-on-exit`. On completion, reopen no-card mode,
read `c0_snapshot_manifest.json`, and bind the resulting three checkpoint
paths and hashes into the G0 dataset-risk configuration. No G0 risk scan is
authorized until that manifest passes.

## C0 snapshot R1 terminal and R2 correction

The first GPU attempt at commit `77aefc7` was terminated by the fatal-log gate
before epoch 0. Model construction, AMP checks, train scanning, and the start of
validation scanning completed, after which PyTorch workers raised
`OSError: AF_UNIX path too long`. The controller reported
`C0_TRAIN failed: fatal_log_signal`; no checkpoint or experimental result was
produced. The R1 run, control, and log roots remain preserved.

The failure was caused by the long data-disk `TMPDIR`, not by the DG-CAIP
method, VOC binding, model definition, or GPU memory. Commit
`f5e223a73d17939402de7613f2152e50a77b07b8` makes three scoped corrections:

- moves the retry to a fresh `-R2` run root;
- requires the temporary root to be at most 48 bytes and binds the reviewed
  retry to `/root/autodl-tmp/t/dg0r2`;
- avoids the instance's unsafe `/usr/bin/shutdown` text wrapper and performs no
  filesystem deletion; terminal shutdown now signals only the detected
  `supervisord` process.

R2 no-card evidence:

- remote focused tests: 14 passed in 3.69 seconds;
- config SHA-256: `a267146adfb7f49ef78d21fe17b393ad8597aaf02a9a60f87217d287d146ac86`;
- preflight evidence SHA-256: `a17b9190eff563d14fd2fe2c45b2730094cde92b6edf01fbcd748865446a15a6`;
- live sparse audit again returned 16,551 total, 6,095 person, and 0 poisoned;
- all five R2 output roots remain absent;
- R1 controller evidence remains present;
- safe shutdown discovery found the current no-card `supervisord` PID without
  changing it;
- no GPU device was visible and no training was launched.

Decision: `pass_r2_no_card_ready_for_single_c0_gpu_refresh`.
