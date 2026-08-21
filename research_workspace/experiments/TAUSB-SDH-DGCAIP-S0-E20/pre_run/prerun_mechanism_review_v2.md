# PRERUN-MECHANISM-02

- Result: blocked
- Decision: do_not_run
- Gated run: `REMOTE-MECHANISM-01`
- Code snapshot: `d81902bae3641a63d9b58664fe6b41fd061d690d`
- Intent: run the matched P1-R/P2-CAIP/P3-DIST/P4-DGCAIP mechanism arms only
  after the passed D0 gate, under a 20-minute hard cap, and freeze P4 state only
  if every preregistered mechanism check passes.
- Code location: the bound config reaches `run_tausb_sdh --stage mechanism`,
  `run_mechanism_pilot`, and then `run_dgcaip_pilot`; `run_mode=mechanism`
  selects the four-arm path and verifies the D0, P1 state, P1 metrics, split,
  and Spec bindings before optimization.
- Parameter data flow: CLI config -> validated `dgcaip` section -> frozen
  calibration and held-out split -> shared adapter initialization -> four arms ->
  `mechanism_metrics.json` decision -> optional `p4_dgcaip_state.pt` only on pass.
- Runtime state: the bound config contains no placeholders and binds D0 SHA256
  `911896f16514639b3b5190d86155f6a48c7711ec1e206828b8e98674114d7539`,
  split SHA256 `9506fb1a981cc5e072dc4176994608b14bb8c39363de615919a2a392fedf4280`,
  and the frozen P1 state/metrics hashes.
- Sink effect: the mechanism runner records fixed Q1/Q4 cohorts, per-arm damage,
  P1 replay, target retention, CICR, pattern change, CGR orthogonality/null
  dimension, fixed protection budget, backtracking, and the preregistered gate.
- Baseline/disable path: P1-R retains current NLA+CGR; P2/P3/P4 share the same
  initial adapter and differ only through the approved CAIP/distribution/ranking
  switches. Focused DG-CAIP and binding tests pass 28/28.
- Local validation: config parsing, bound-hash checks, no-placeholder assertion,
  and 28 focused tests passed on the reviewed snapshot content.
- Minimal probe: D0 itself completed on the exact parent snapshot and passed all
  locator gates; this is diagnostic evidence only and does not substitute for
  the four-arm mechanism run.
- Run command binding: **blocked**. The exact commit contains only
  `dgcaip_d0_controller_v1.sh` and `dgcaip_d0_tmux_launch_v1.sh`. Both hard-code
  D0 paths/schema/session and the controller explicitly asserts
  `dgcaip.run_mode == "d0"`. No reviewed mechanism-specific tmux command exists.
- Experiment validity: VOC20, target id 14, seed 0, frozen secret/surrogate/P1/D0
  hashes, 16 calibration batches, 24 held-out batches, 8 steps, and no EOT/JND
  are correctly frozen in the bound config.
- Output non-overwrite: the config uses a unique mechanism artifact root, but no
  reviewed launcher currently checks that artifact/control/cache/tmp/log/session
  paths are absent before handoff.
- Recoverability/secrecy: **blocked**. There is no mechanism-specific outer
  timeout, tmux handoff, controller status, all-terminal shutdown wrapper, or
  prelaunch-failure record in the exact commit. No credentials are present.
- Blockers: add the minimal mechanism-specific controller and tmux launcher,
  enforce the same 1200-second outer timeout and data-disk/non-overwrite rules as
  D0, validate `run_mode=mechanism`, and request a new exact snapshot review.
- Validation gaps: no mechanism arms, fresh victim, or AP50 evidence exists.
