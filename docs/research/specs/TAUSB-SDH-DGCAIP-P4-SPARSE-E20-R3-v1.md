# TAUSB-SDH-DGCAIP-P4-SPARSE-E20-R3-v1

Status: approved by the user on 2026-08-30.

## Scope

This execution-only revision fixes the output-root binding failure observed in
R2. R2 isolated its controller, logs, binding, and victim roots, but the
mechanism YAML still bound `runtime.artifact_root` to the preserved R1 root.
The mechanism therefore stopped before computation with `FileExistsError`.

R3 uses one fresh root consistently in the mechanism YAML, one-boot command,
binder, sparse controller, victim roots, comparison root, and logs. The
one-boot controller now fails closed before creating run artifacts whenever
the YAML artifact root and `--mechanism-root` differ.

The wrapper also acquires an atomic R3 launch lock before installing its
shutdown trap and refuses to overwrite an existing outer log. A duplicate
launch therefore cannot truncate evidence or shut down the active R3 process.

## Frozen scientific contract

The carrier, P4 mechanism, state-integrity and scientific gates, VOC inputs,
person target, C0/M1 construction, fresh-victim training, E20 horizon, seed,
thresholds, AP50 reporting, claim boundary, and historical P1 comparison are
unchanged from `TAUSB-SDH-DGCAIP-P4-SPARSE-E20-v1`.

## Isolation and cost

- R1 and R2 evidence is retained without deletion or reuse.
- All R3 run roots begin with
  `/root/autodl-tmp/tausb-dgcaip-runs/TAUSB-SDH-DGCAIP-S0-P4-SPARSE-E20-R3`.
- The R3 outer log is
  `/root/autodl-tmp/tausb-dgcaip-wrapper-logs/TAUSB-SDH-DGCAIP-S0-P4-SPARSE-E20-R3.outer.log`.
- Mechanism wall cap remains 20 minutes.
- Total one-boot wall cap remains 2 hours.
- Every terminal outcome retains evidence and invokes the unconditional
  shutdown trap.

R3 may start only after focused tests, exact config/wrapper binding checks,
frozen input hashes, fresh R3 roots, sufficient data-disk capacity, an exact
clean checkout, and one visible idle GPU all pass.
