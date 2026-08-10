# TAUSB-SIRC-MALC-CGR-MAP50-S0 mechanism gate

## Decision

- Spec: `TAUSB-SIRC-MALC-CGR-MAP50-v2`
- Run: `mechanism-fcf26cc-seed0`
- Branch: `codex/tausb-sirc-malc-cgr-map50-v2`
- Reviewed code: `fcf26cc24dc7e6943234cc3cdf7943fd957cb6cc`
- Gate result: **FAIL**
- `allow_fresh_victim`: **false**
- Consequence: C0/M1 and fresh-victim AP50 training remain forbidden by the pre-registered stop condition.
- Claim boundary: held-out surrogate mechanism evidence only; this result is neither fresh-victim UE evidence nor a robustness result.

The run completed both matched arms and stopped normally at the scientific mechanism gate. The wrapper's final assertion and exit code 1 are an intentional enforcement of the failed gate, not a new runtime defect. The cost guard then requested AutoDL shutdown.

## Evidence integrity

- The pull manifest selected ten exact small mechanism files; all ten transferred and `missing_required=[]`.
- `transfer-report.json` records a SHA-256 hash for every selected file.
- The resolved split hash is `e2542517af00830147117582d69ff15a62fbeae1f8583bf0c9d01fbff120cae1` in both arms and calibration evidence.
- Input audit passed source-contamination checks and matched the frozen source, semantic-bank, C2LM-basis and surrogate-checkpoint hashes.
- Both A0 and A1 completed 40 optimization steps and 24 held-out batches.
- Control status: `failed / mechanism_gate_check / formal_pipeline_exit_1_shutdown_requested`; the formal mechanism status itself is `stopped / mechanism_gate`, with no frozen carrier.

Primary evidence:

- `remote_artifacts/mechanism_report.json`
- `remote_artifacts/arms/A0_metrics.json`
- `remote_artifacts/arms/A1_metrics.json`
- `remote_artifacts/arms/A0_diagnostics.json`
- `remote_artifacts/arms/A1_diagnostics.json`
- `remote_artifacts/malc_calibration.json`
- `remote_artifacts/status.json`
- `remote_artifacts/transfer-report.json`
- `remote_artifacts/control/formal-seed0-fcf26cc.log`
- `remote_artifacts/control/cost-guard-status-fcf26cc.json`

## Pre-registered gate audit

| Signal | A0 | A1 | Required | Result |
|---|---:|---:|---:|---|
| Residual cosine median | 0.137720 | 0.142559 | A1-A0 >= 0.10 | **FAIL**; gain 0.004839 |
| Residual cosine Q25 | -0.050517 | -0.047634 | A1 > 0 | **FAIL** |
| Log-energy MAD | 0.521054 | 0.519136 | A1 <= 0.90*A0 = 0.468948 | **FAIL**; A1/A0 = 0.99632 |
| Valid-instance coverage | 0.996212 | 0.996212 | A1 >= 0.80 | PASS |
| Zero-norm ratio | 0.000000 | 0.000000 | A1 <= 0.20 | PASS |
| Floor-pass ratio | 0.970651 | 0.970651 | A1 >= 0.80 | PASS |
| Valid scales at 0.80 | 2/3 | 2/3 | >=2/3 | PASS |
| CGR max projected row dot | 8.20e-08 | 2.09e-07 | A1 <= 1e-5 | PASS |
| CGR attack retention median | 0.962688 | 0.969713 | A1 >= 0.20 | PASS |
| CGR repair+skip ratio | 0.083333 | 0.083333 | A1 < 0.50 | PASS |
| Non-target/target residual energy | 0.283806 | 0.283250 | A1 <= 1.25*A0 | no failure |
| Box residual energy | 0.498257 | 0.499073 | A1 <= 1.25*A0 | no failure |

The mechanism failed all three efficacy signals while passing coverage, non-zero energy, scale validity, CGR feasibility and leakage safeguards.

## Key finding

The first bad scientific boundary is not CGR saturation or missing assignments. It is that enabling MALC does not create a materially different carrier trajectory from the matched MALC-off arm.

Across the 40 paired optimization steps:

- Mean absolute A0/A1 difference in `easy_cls_loss`: `0.001708`.
- Mean absolute A0/A1 difference in the observed `malc_loss`: `0.001813`.
- Mean absolute A0/A1 difference in `rms_loss`: `0.0000482`.
- A0 and A1 use nearly the same CGR modes: both have 33 `projected_target` and 3 unprojected `target` steps; their repair/skip counts differ by only one step.
- A1 keeps 96.97% of the combined target-gradient norm after CGR, so an exhausted CGR null space is not supported by the evidence.
- Per-group cosine changes are all approximately zero: cooccur `+0.000463`, person-only `-0.001341`, large `+0.000886`, medium `-0.000593`, small `+0.000473`.

Thus MALC is numerically active in the scalar objective, but it is not producing the intended held-out residual concentration.

## Root-cause assessment

### Established by this run

1. The repaired implementation completed; the earlier live-autograd OOM is no longer the active failure.
2. TAL/PAG coverage, residual energy and two FPN scale-validity gates are healthy.
3. The non-target constraints do not globally erase the composite target update.
4. The MALC-on arm is almost indistinguishable from MALC-off on both optimization diagnostics and held-out signatures.

### Strong design-level inference

The current calibration equalizes only component gradient **norms**. It does not measure the quantities central to the hypothesis: MALC gradient direction agreement across batches, conflict with `easy_cls`/RMS, or survival of the MALC component after CGR projection. The recorded median norms (`easy_cls=0.21434`, `malc=0.23589`, `rms=0.01886`) produce weights (`malc=0.90865`, `rms=10.0`), but equal norm does not imply a stable shared descent direction.

The prototype calibration also accepts one normalized mean direction per scale whenever its norm is merely non-zero. It records counts, but not resultant length, angular dispersion or bootstrap stability. A1's held-out cosine Q25 remains negative, and the single global prototype helps some groups by less than 0.001 while slightly hurting others. This is consistent with a multimodal or batch-conflicting residual field being compressed into one weak global direction.

### Not yet isolated

The SIRC carrier uses `1+tanh(theta)` to positively modulate 16 fixed semantic bases shared by four variants. This may restrict the signed directions MALC can realize, but the current artifacts do not separate carrier-Jacobian expressivity from gradient conflict. It must not yet be called the root cause.

## Smallest discriminating next step

Do **not** rerun C0/M1 or another full 40-step mechanism experiment. First add a calibration-only gradient-geometry audit, without changing the carrier or loss:

1. Record per-scale prototype resultant length `||mean(normalize(r))||` and leave-one-batch/bootstrapped prototype angular stability.
2. Record per-batch pairwise cosines among `g_easy`, `g_malc` and `g_rms`, plus cross-batch cosine of `g_malc`.
3. Apply the active CGR projector separately to each component and record `||P g_k||/||g_k||`; the existing combined-gradient retention cannot show whether CGR selectively removes MALC.
4. Record the A0/A1 coefficient distance, rendered-pattern distance and update cosine after each matched batch.

This single probe yields a decision:

- Low prototype resultant length or conflicting cross-batch MALC gradients: replace the single global pull with a conditional/mixture or pairwise concentration objective; do not tune `lambda_malc`.
- Healthy MALC gradient geometry but low component-wise post-CGR survival: calibrate and combine gradients in the CGR feasible subspace.
- Healthy pre/post-CGR MALC geometry but negligible pattern separation: revise the carrier parameterization, likely allowing zero-centred signed basis coefficients or limited variant-specific degrees of freedom.

Changing the weight, learning rate, carrier and prototype together would not identify the mechanism and is therefore not recommended.
