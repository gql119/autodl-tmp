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

`pass_l0_ready_for_g0_binding`
