# TAUSB strict-route v2 G1 evidence analysis

## Decision

The run completed normally, but the scientific gate failed. G2 and M1 remain
closed. This is not a CUDA, timeout, determinism, storage, or controller failure.

- Spec: `TAUSB-SDH-DGCAIP-STRICT-ROUTE-v2`
- Execution commit: `7978182b620707525d84870c73b49ef5bd923dba`
- Controller status: `completed`
- Mechanism exit code: `0`
- GPU process observed: `true`
- Controller/mechanism elapsed: `35.17 s` / `29.51 s`
- Fatal scan: zero matches for traceback, OOM, runtime error, NaN or Inf
- Automatic shutdown: observed immediately after the terminal controller state
- Candidate state retained even though the gate failed

## Integrity

The locally pulled evidence matches the controller-recorded hashes:

- `mechanism_metrics.json`:
  `056e505cec63edd742e7ee7e2a2d7e21421311262357d8b4c45a2807ce3a9396`
- `backtracking_trace.json`:
  `e88966d9b7642e16ef9065785e1de4892b3ae1e6fc7bce30088243bd43909e94`
- `p5_dataset_strict_state.pt`:
  `b0cc8c20bab0a381f990e458eacdda31cb03f98ca1cd462cd3e34eefd01c5bd5`

All evidence is stored under the sibling `remote_artifacts` directory.

## Gate result

Ten of twelve checks passed. The failed checks were:

1. `backtrack_skip`: only 1/8 routed updates was accepted, giving a skip ratio
   of `0.875`, above the frozen `<0.70` requirement.
2. `final_target_progress`: the sole accepted step recorded
   `0.5999999642`. The router correctly accepts values above
   `0.60 - 1e-6`, but the final gate uses an exact `>=0.60` comparison.
   This is a gate-tolerance mismatch of only `3.58e-8`. Fixing it alone would
   not make the run pass because `backtrack_skip` still fails.

The passing checks were input-bank binding, replay binding, at least one update,
safe orthogonality, violated non-worsening direction, nonzero null space, attack
retention, finiteness, adapter change and frozen-module integrity.

## Per-step routing evidence

| Step | Route feasible | Accepted | Attempts | Rank | Null dim | Attack retention | Target progress | Safe max dot | Violated min dot |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | yes | no | 6 | 3 | 3520 | 0.9015 | 0.6000 | 5.96e-8 | 0.00 |
| 1 | yes | no | 6 | 2 | 3521 | 0.9706 | 0.6000 | 2.98e-8 | -1.19e-7 |
| 2 | yes | no | 6 | 0 | 3523 | 1.0000 | 0.7156 | 0.00 | -5.96e-8 |
| 3 | yes | no | 6 | 4 | 3519 | 0.8574 | 0.6000 | 4.77e-7 | 1.19e-7 |
| 4 | yes | yes | 1 | 5 | 3518 | 0.7771 | 0.599999964 | 1.49e-7 | -1.19e-7 |
| 5 | yes | no | 6 | 0 | 3523 | 1.0000 | 0.8039 | 0.00 | -2.38e-7 |
| 6 | yes | no | 6 | 2 | 3521 | 0.9274 | 0.7244 | 1.19e-7 | -1.19e-7 |
| 7 | yes | no | 6 | 5 | 3518 | 0.8785 | 0.6000 | 2.09e-7 | -8.94e-8 |

All eight linear routes were feasible. Safe dots stayed below `1e-5`, violated
dots stayed above `-1e-6`, and target progress stayed within the router's frozen
tolerance. Therefore strict-route v2 fixed the R3 linear-feasibility failure.

## Why seven nonlinear updates were rejected

Backtracking tried learning rates from `5e-4` through `1.5625e-5`. At the final
attempts of the seven rejected steps, violations comprised:

- alignment: 11, maximum positive margin `1.62e-4`
- IoU: 14, maximum positive margin `2.96e-5`
- probability: 2, maximum positive margin `4.71e-7`
- JS: 1, maximum positive margin `3.13e-8`

The primary implementation mismatch is that the linear router receives one
combined `NLA + DG-CAIP` gradient row per snapshot/class, while nonlinear
backtracking checks probability, IoU, alignment and JS independently. A
non-worsening direction for the sum does not imply non-worsening of each
component. The observed positive margins generally decrease approximately
linearly as the step size is halved, which is consistent with unmatched
first-order constraints rather than only second-order curvature or float noise.

Increasing the number of backtracks or broadly loosening tolerances would hide
this mismatch and is not the recommended first repair.

## Scientific observations

- The update geometry retains useful target signal: attack retention ranges
  from `0.7771` to `1.0000` and target progress from `0.6000` to `0.8039`.
- DLFC cosine remains high (`0.9929` median), while CICR remains weak
  (`0.3187` median). Only one update was accepted, so this run cannot judge
  whether the final carrier produces target-class unlearnability.
- Non-target probability-drop macro is small (`2.37e-5`) in this mechanism
  probe, but it is not victim AP50 evidence and must not be presented as final
  non-target preservation.

## Required next revision

Before another GPU run:

1. Make the routed gradient rows and nonlinear acceptance metrics identical in
   granularity: snapshot / class / family. NLA must remain an explicit row or an
   explicitly audited metric rather than being silently mixed into a row whose
   backtracking check is different.
2. Keep safe rows as equalities and violated rows as non-worsening half-spaces,
   but apply them to the same component losses used by backtracking.
3. Make the final target-progress gate use the router's frozen `1e-6`
   post-cast tolerance.
4. Add tests that fail when a combined class loss is non-increasing while an
   individual protected component increases.
5. Run a no-card replay/unit audit before authorizing a new single G1. Do not
   proceed to G2 or victim training from this failed candidate.
