# TAUSB-SDH-DGCAIP-P4-SPARSE-E20-R2-v1

Status: approved by the user on 2026-08-30.

## Scope

This revision permits one retry of the approved P4 sparse paired E20 run after
the first attempt failed before completing its first held-out summary. The
failure was an execution-only PyTorch 2.0 incompatibility: subtraction from a
boolean union-support mask. Commit `9d7b3ac13da0dfa7b1f76660bc4c8e1ef3b5e603`
replaces that subtraction with logical negation, adds a reproducing regression
test, and prepares the already-bound cache directories.

## Frozen scientific contract

The carrier, P4 mechanism, state-integrity and scientific gates, VOC inputs,
person target, C0/M1 construction, fresh-victim training, E20 horizon, seed,
thresholds, AP50 reporting, claim boundary, and historical P1 comparison remain
identical to `TAUSB-SDH-DGCAIP-P4-SPARSE-E20-v1`.

## Retry isolation and cost

- The original failed roots and logs are preserved without deletion or reuse.
- The retry uses only roots beginning with
  `/root/autodl-tmp/tausb-dgcaip-runs/TAUSB-SDH-DGCAIP-S0-P4-SPARSE-E20-R2`.
- Its outer log is
  `/root/autodl-tmp/tausb-dgcaip-wrapper-logs/TAUSB-SDH-DGCAIP-S0-P4-SPARSE-E20-R2.outer.log`.
- Mechanism wall cap remains 20 minutes.
- Total one-boot wall cap remains 2 hours.
- Every terminal outcome retains evidence and invokes the unconditional
  shutdown trap.

The retry may start only after an exact clean checkout, focused no-card tests,
frozen input hashes, R2 fresh roots, data-disk capacity, wrapper/config hashes,
and one visible idle GPU all pass.
