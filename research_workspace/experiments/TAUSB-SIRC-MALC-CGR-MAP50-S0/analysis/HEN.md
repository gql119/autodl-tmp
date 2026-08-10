# TAUSB-SIRC-MALC-CGR-MAP50-S0 H->E->N

## Hypothesis

Detector-native, assignment-aware concentration of person classification-tower residuals should make a shared SIRC carrier produce a stable shortcut across appearance, scale and context, while CGR preserves non-target foreground responses.

## Evidence

- The matched A0/A1 run on reviewed commit `fcf26cc24dc7e6943234cc3cdf7943fd957cb6cc` completed mechanism evaluation without a runtime exception.
- Required mechanism artifacts were pulled exactly and all ten required files passed local presence and SHA-256 verification.
- A1 failed the three mechanism-efficacy signals:
  - cosine-median gain `0.004839`, versus required `0.10`;
  - cosine Q25 `-0.047634`, versus required `>0`;
  - log-energy MAD ratio `0.99632`, versus required `<=0.90`.
- Coverage, non-zero residual energy, two-scale validity, CGR orthogonality/retention and both leakage safeguards passed.
- A0/A1 component diagnostics differ only marginally over 40 matched steps, showing that MALC does not materially separate the optimized carrier from the MALC-off baseline.
- No A1 carrier was frozen. C0/M1, poisoned-data generation, victim training and AP50 evaluation did not run.

Full gate table and provenance: `analysis/mechanism_gate.md`.

## Interpretation

Outcome: **mechanism FAIL, fresh-victim evidence unavailable by design**.

The evidence rejects the present implementation of the single frozen MALC prototype as an effective residual-concentration mechanism under the approved carrier and CGR route. It does not reject the broader idea of a robust semantic carrier, and it does not show whether a victim would be selectively unlearnable because the pre-registered gate correctly prevented that expensive experiment.

The key design gap is directional: the method hypothesizes cross-sample gradient/residual agreement but calibrates only gradient magnitudes. The artifacts therefore support a focused gradient-geometry diagnosis, not another parameter sweep.

## Next

Freeze a follow-up calibration-only diagnostic that records prototype angular stability, cross-batch/component gradient cosines, per-component post-CGR survival, and matched A0/A1 coefficient/pattern separation. Use those measurements to choose exactly one subsequent change:

- conditional/mixture residual concentration;
- null-space-aware MALC gradient combination; or
- a more expressive signed carrier parameterization.

Do not start C0/M1 and do not claim AP50, robustness, transferability, MALC fresh-victim causality or SOTA from this mechanism-only result.
