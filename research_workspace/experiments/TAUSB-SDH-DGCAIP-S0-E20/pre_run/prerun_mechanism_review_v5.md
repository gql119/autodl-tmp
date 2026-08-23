# PRERUN-MECHANISM-05

- Result: blocked
- Decision: do_not_run
- Gated run: `REMOTE-MECHANISM-02`
- Code snapshot: `84536f9ea486b0e67355ab74619188ce22229bf2`
- Intent: rerun the matched P1-R/P2-CAIP/P3-DIST/P4-DGCAIP mechanism arms
  after the finite float32 divergence-roundoff repair, under the unchanged
  20-minute cap and without overwriting the preserved R1 failure evidence.
- Code location: the exact snapshot contains the source-level divergence fix,
  its regression tests, the bound mechanism config, four-arm runner, controller,
  and R2 tmux launcher.
- Parameter data flow: bound config -> validated D0/P1/split/input hashes ->
  shared calibration and held-out batches -> four arms -> mechanism metrics ->
  optional P4 state only on gate pass.
- Runtime state: clean detached AutoDL checkout at the exact pushed commit;
  remote branch SHA matches, and the checkout has zero dirty paths.
- Sink effect: config validation passes and the repaired divergence reaches the
  active DGCAIP weighting path; the remote CPU probe returns JS/KL minima of 0,
  zero negative entries, and finite gradients.
- Baseline/disable path: P1-R remains the frozen NLA+CGR reference; no loss,
  dataset, model, arm switch, or scientific gate changed.
- Local validation: 21 focused and 65 related tests passed for the numerical
  repair; both controller scripts pass Bash syntax checks.
- Minimal probe: remote exact-checkout config parse, Bash syntax, numerical
  probe, six required-input existence checks, disk check, and R2 path-absence
  checks passed.
- Run command binding: blocked. The launcher binds ARTIFACT_ROOT to
  `/root/autodl-tmp/tausb-dgcaip-runs/TAUSB-SDH-DGCAIP-S0-R2-MECHANISM`,
  but the exact config still binds runtime.artifact_root to the R1 path.
- Experiment validity: VOC20, person id 14, seed 0, frozen mechanism settings,
  no EOT/JND, and original mechanism gates remain unchanged.
- Output non-overwrite: all six R2 formal paths are absent; R1 evidence remains
  present and is not targeted.
- Recoverability/secrecy: tmux, 1,200 s outer timeout, terminal status,
  data-disk-only growth, and all-terminal shutdown remain present; no credential
  was printed or persisted.
- Blockers: launcher/config artifact-root mismatch. Fix both the config root and
  pinned normalized config SHA, commit and push, then run a new exact review.
- Validation gaps: no mechanism arm or fresh-victim/AP50 result exists; GPU was
  not started during this blocked review.
