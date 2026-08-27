# TAUSB-SDH-DGCAIP R3 pre-run implementation review 01

- Review date: 2026-08-27
- SpecID: `TAUSB-SDH-DGCAIP-R3-DIAG-v1`
- ExpID: `TAUSB-SDH-DGCAIP-S0-R3-DIAG`
- Gated run: `REMOTE-DIAG-01`
- Branch: `codex/tausb-sdh-dgcaip-r3-diag-v1`
- Code snapshot: `8a54f74e094c7a15fe4bc487b29ea8712bfe426e`
- Config SHA256: `3d213af011234cdf08ef0b54d78c377db18ddaf83903d771057b84f418839613`
- Local implementation result: pass
- Overall pre-run result: pass
- Decision: ready for the single capped GPU diagnostic after the user enables GPU

## Independent packet review

### Intent and claim boundary

The snapshot performs one diagnostic-only mechanism run. It records nonlinear
P2/P4 constraint rejection and repeats P1 twice in one process. It does not
materialize a poisoned dataset, train a victim, evaluate AP50, run E20/E200, or
promote a P4 state. A diagnostic label is not an effectiveness claim.

### Parameter and data flow

```text
tausb_sdh_dgcaip_r3_diag_v1.yaml
  -> validate_sdh_experiment_config
  -> run_mechanism_pilot
  -> run_dgcaip_pilot
  -> P1-A / P1-B / P2-CAIP / P4-DGCAIP
  -> active legacy backtracking decision
  -> default-off observational trace
  -> H1 rejection attribution + H2 same-process replay
  -> four JSON diagnostic artifacts
```

Frozen inputs remain seed 0, VOC20, person id 14, image size 640, 16 calibration
samples, 24 held-out samples, batch 4, eight steps, `eps=16/255`, no EOT/JND,
and the R2 D0/P1/hiding source hashes.

### Feature-off and baseline path

- The legacy `DGCAIP_ARMS` remains P1-R/P2/P3/P4.
- `record_trace` defaults to false and returns an empty trace.
- A deterministic test compares traced and untraced candidate, acceptance,
  attempts, step size, values, and status exactly.
- R3 is selected only by its new SpecID and
  `dgcaip.r3_diagnostics.enabled=true`.

### Active sink and artifacts

The active decision remains the original heterogeneous-limit comparison.
Diagnostics observe the evaluated values and independently recompute agreement.
The R3 path writes:

- `r3_diag/backtracking_trace.json`;
- `r3_diag/p1_same_process_replay.json`;
- `r3_diag/rejection_attribution.json`;
- `r3_diag/mechanism_metrics.json`.

It returns before the legacy `p4_dgcaip_state.pt` save block.

### Cost, storage, and recovery

- Controller hard cap: 600 seconds, with remaining time calculated after input
  validation.
- Unique artifact/control/cache/tmp/log roots use the AutoDL data disk.
- All-terminal controller trap requests shutdown.
- Prelaunch failure trap records evidence and requests shutdown.
- Run is launched in tmux from a clean detached exact-commit checkout.
- R2 artifact and control roots are not targeted.

### Local evidence

```text
compileall: exit 0
config validation: TAUSB-SDH-DGCAIP-R3-DIAG-v1 r3_diag 600
focused and related pytest: 41 passed in 3.66s
controller Bash syntax: exit 0
launcher Bash syntax: exit 0
stray patch-marker argument scan: pass
git diff --check: exit 0
committed config blob SHA256:
3d213af011234cdf08ef0b54d78c377db18ddaf83903d771057b84f418839613
code/scripts differ from reviewed commit: no
```

## Exact reviewed launch

```bash
EXECUTION_COMMIT=8a54f74e094c7a15fe4bc487b29ea8712bfe426e \
  /bin/bash research_workspace/experiments/TAUSB-SDH-DGCAIP-S0-R3-DIAG/pre_run/r3_diag_tmux_launch_v1.sh
```

## Remote no-card evidence

Static pre-run review found and repaired stray patch-marker arguments in both
shell scripts. Therefore `02d1069cf551880f641758a5f9d794ad6b397871` is
superseded and must not be executed. The repaired snapshot
`8a54f74e094c7a15fe4bc487b29ea8712bfe426e` was pushed by explicit authorization.
GitHub and the fetched AutoDL remote-tracking ref both point exactly to this SHA;
the committed config blob SHA256 is
`3d213af011234cdf08ef0b54d78c377db18ddaf83903d771057b84f418839613`.

The no-card audit verified:

- 16,551 train images, 16,551 train labels, and 6,095 person images;
- both frozen image/label manifest hashes;
- surrogate, P1 state, P1 metrics, hiding checkpoint, hiding metrics, and D0
  hashes;
- Python, tmux, shutdown, and the exact Git commit object;
- all seven R3 checkout/artifact/control/cache/tmp/log paths absent;
- no R3 tmux session and no GPU device in no-card mode;
- 7,116,300,288 free bytes on the data disk versus a 1.21 GiB tracked checkout.

Execution gate:

1. wait for explicit confirmation that the GPU instance is enabled;
2. launch only the reviewed command at the exact commit;
3. preserve the 600-second hard cap and all-terminal shutdown behavior;
4. do not run P3, victim training, AP50, E20, or E200.
