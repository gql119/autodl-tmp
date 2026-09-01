# G1 R3 H→E→N analysis

## Hypothesis

The frozen dataset-ranked strict-CGR route can preserve the target attack while
keeping the complete update in the safe non-target null space, actively repair
already violated non-target constraints, and accept enough updates under a
repair norm budget of 0.25 times the target-gradient norm.

## Evidence

- Exact execution commit: `d861dd461094982ddc77c2c01af7999c4f470fbb`.
- Controller status: `completed`; mechanism exit code 0; GPU observed; no fatal
  runtime signature; 35.16 seconds total.
- All G0 risk-bank, replay, e1/e5/e20 snapshot, P1 and frozen-module bindings
  passed. All scalar and tensor checks were finite. Perturbation outside the
  person support was exactly zero.
- Projected target retention was not the blocker: 0.7771 minimum, 0.9026
  median, and 1.0 maximum. Safe rank was 0--5 and the null dimension remained
  3,518--3,523.
- The required repair norm was 0.3166--0.7575 of the target-gradient norm
  (median 0.5984), above the frozen 0.25 budget on every step. Thus all eight
  routes were `skip_infeasible_constraints`, all selected final gradients were
  zero, no nonlinear candidate was evaluated, and 0/8 updates were accepted.
- The raw candidate repair floor was reached within 1e-6, but several raw
  candidates had safe-row residuals above the 1e-5 tolerance; the observed
  maximum was 4.37e-5.
- Local copies match the remote SHA-256 values for controller status, result,
  preflight, metrics, trace and candidate state. The raw artifacts remain under
  `remote_artifacts/` and are deliberately not promoted into a passing state.

## Conclusion

The implementation is now runnable end to end, but the current strict-CGR
geometry and frozen feasibility hyperparameters reject every update. This is a
scientific/design gate failure, not another code bug. It provides no evidence
that the revised carrier improves a fresh victim and does not authorize G2 or
M1.

## Next decision

Freeze a small, separate feasibility-calibration Spec before another GPU run.
The diagnostic should use disjoint fixed calibration and validation replay
slots and pre-register a small grid over repair-floor strength and repair norm
budget. Selection must choose the lowest-protection-cost setting that attains a
nonzero accepted-update rate while retaining the target direction; validation
must still enforce final safe-row dots, nonlinear non-worsening, finite values,
and a skip ratio below the existing 0.70 gate. Add explicit final-update target
cosine and raw safe-residual diagnostics. Do not change the carrier, ranking,
snapshots, losses, dataset, or victim protocol in that calibration.

Only after the revised route passes on held-out replay slots should the project
rerun the formal eight-step G1 and then consider the three-epoch G2 audit.
