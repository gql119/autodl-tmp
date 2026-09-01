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

## C0 snapshot R2 terminal evidence

The corrected C0 refresh completed on the RTX 4090D and shut the instance down
through the reviewed `supervisord` signal path. The controller status is
`completed`, all 20 epochs are present, `fatal_count=0`, and the elapsed time was
1,262.00 seconds. The clean sparse input remained 16,551 total training images,
6,095 person images, and zero poisoned images. The fresh-victim tensor hash also
matched the frozen expected value.

The snapshot manifest passed and binds the exact post-epoch states required by
the Spec:

- e1: internal epoch 0, SHA-256
  `6ebacf59d7fa27ae8d30bb86571d5f089392e19d52ba9ffd7fd204faa70c5ae1`;
- e5: internal epoch 4, SHA-256
  `cfaf454563e7ac81676468ec09fb08a94718a9902c5ee7057ee3db0d63202fc4`;
- e20: internal epoch 19, SHA-256
  `e660ed4b2f36e8b866f89a4f88a02e3d3a7eed6f2727f99573cc3c4d8bfaad53`.

The controller-status SHA-256 is
`8c19fee403622a84ceaee6b07b52e65e73053fc313ab948d8f8329c9f7b18288`,
and the snapshot-manifest SHA-256 is
`718a971b9cd7c09c24909f1e49e8d4c4e9a6de71b045e4c9e8c9ab1e3058b6fc`.

## G0 risk-scan binding and no-card gate

Commit `cc0f9b42e265100a835985bfc4ab3e95411470dd` adds the bound G0
controller and configuration. It freezes the three C0 snapshots above, the C0
manifest and execution commit, and the existing P1 state SHA-256
`2e102026a9356116de38acb1f5056bf5728afcd453e3447b516d4222f4d70b81`.
The scan is limited to `dataset_risk_scan`, has a 60-minute method budget and a
65-minute controller hard cap, writes only to the AutoDL data disk, preserves a
scientifically failed coverage decision, and uses the reviewed safe shutdown
path.

Remote no-card evidence:

- focused G0 plus adjacent dataset-risk, strict-CGR, proxy-agreement, C0 and
  config tests: 49 passed in 28.53 seconds;
- exact detached checkout: `cc0f9b42e265100a835985bfc4ab3e95411470dd`,
  clean tracked worktree;
- config SHA-256:
  `45aef454982548c1a9b9b954342cd4c794ef2ca8a4172e882740c57df1a95c66`;
- preflight evidence SHA-256:
  `fa705b97645a3cbd4a60aa920f0d723f3750fb828738f6acc6cf7bacf75852e9`;
- live data-disk free space: 9,024,917,504 bytes;
- all five G0 output roots remained absent after preflight;
- `nvidia-smi` exposed no device, so no GPU scan was launched.

Current decision: `pass_g0_risk_no_card_ready_for_single_gpu_scan`.

## Current next gate

Enable GPU only for `G0-RISK-R1` and run the reviewed controller at commit
`cc0f9b42e265100a835985bfc4ab3e95411470dd` with `--shutdown-on-exit`.
The controller will scan person-cooccurrence non-target instances under the
frozen e1/e5/e20 teachers, write the dataset-level risk bank, replay manifest,
raw records, coverage decision and controller evidence, then shut down on every
terminal exit. It does not generate a new poison dataset or train a victim.

## G0 dataset-risk terminal evidence

G0 completed successfully at execution commit
`cc0f9b42e265100a835985bfc4ab3e95411470dd` and shut down automatically.
The guarded child exited 0 after 360.47 seconds; the scientific scan itself
reported 353.27 seconds. No Traceback, OOM, NaN/Inf, fatal signal, or hard-cap
event was observed.

The complete person-cooccurrence subset contained 4,380 images and 8,337
non-target ground-truth instances. All 8,337 instances were observed under all
three clean-victim snapshots, giving both overall coverage and stable-snapshot
coverage of 1.0. The coverage decision passed. Class-wise top-25% selection
produced 2,090 high-risk instances in total; the 32 replay slots retain the
frozen 50/50 high-risk/uniform protocol. The snapshot producing the worst risk
was e1 for 3,091 instances, e5 for 2,597, and e20 for 2,649, which supports the
need for the approved multi-snapshot worst-risk aggregation.

Independent local hash verification after pulling the minimum evidence set:

- risk manifest:
  `543c632810e498daf147d6687e8dc6ac7c50fcbac7404170dabd87e9e2246a62`;
- risk bank file:
  `21cf001ed69b030a6dce1a7e9ea67b07de45f0f41189a9c828fe9b9e3488fabe`;
- risk bank canonical payload:
  `3dcc755fc7629cc5d2b37bd7b6931088001bf0ca0d3976343d7420d4236eb5fc`;
- replay manifest:
  `e5dd31cac06d038f4fc305970a9a60a2e2f34b3ae61e55af7405568fbbb7e457`;
- raw records:
  `ec509b7e31a236549e448aab473aa3a570955b951ecba85a21d894845f04c3bb`;
- controller status:
  `ffbd210e9fe889f584d4fa6a4c84384bcf518382818bfa48031bb660a0c63d3b`.

This gate establishes complete, deterministic proxy-risk coverage; it is not
an AP50 or non-target-retention result.

## G1 strict-mechanism no-card gate

Commit `216ded9217b98c3f5661eaa355d3bbc8e3c42686` binds the single G0
terminal chain above into one `P5-DATASET-STRICT` arm. The controller verifies
the G0 controller, risk manifest, file and canonical bank hashes, replay hash
and 32-slot count, e1/e5/e20 checkpoint files, P1 state, exact clean worktree,
data-disk containment, free space and fresh output roots before exposing GPU.
It uses eight optimization steps, an 1,100-second method budget, a 20-minute
controller hard cap, and the reviewed safe terminal shutdown path. A failed
scientific decision is preserved as a result rather than converted into a
runtime failure.

No-card evidence:

- focused G1/G0/dataset-risk/strict-route/proxy suite: 58 passed in 6.23 seconds;
- exact detached checkout:
  `216ded9217b98c3f5661eaa355d3bbc8e3c42686`, clean tracked worktree;
- config SHA-256:
  `a7ebe1424b58f7797654aed16f66363fe3f3fa26dada87134e11d14d6e059003`;
- preflight evidence SHA-256:
  `582d12370a16b7d5abf43f99623c86114959988fc4bc69dc1c2aebcf83827e74`;
- live data-disk free space: 9,014,054,912 bytes;
- all five G1 output roots remained absent after preflight;
- no GPU device was exposed and no mechanism optimization was started.

Current decision: `pass_g1_strict_no_card_ready_for_single_gpu_run`.

## Current next gate after G0

Enable GPU only for `G1-STRICT-R1` and run the reviewed controller at commit
`216ded9217b98c3f5661eaa355d3bbc8e3c42686` with `--shutdown-on-exit`.
The only new computation is the eight-step dataset-ranked strict-CGR mechanism.
Proceed to the 3-epoch short-victim G2 audit only if all recorded G1 checks pass.
