# TAUSB-SDH-DGCAIP-R4-D0-BINDING-FIX-v1

## 1. Status

- SpecID: `TAUSB-SDH-DGCAIP-R4-D0-BINDING-FIX-v1`
- Status: `approved`
- Approved: `2026-08-27` (explicit user approval in the active task)
- Scope: provenance-binding repair plus one diagnostic mechanism rerun
- Parent failure: `TAUSB-SDH-DGCAIP-S0-R3-DIAG`
- Target class: `person` (`class_id=14`)

This Spec is approved for local implementation and validation. Push, remote
checkout, and GPU execution remain separately gated.

## 2. Confirmed R3 Failure

R3 stopped after 2.81 seconds with
`DG-CAIP D0 report SpecID mismatch.` The controller returned exit code 1 and
requested shutdown. No P1-A/P1-B/P2/P4 diagnostic output was produced.

The frozen D0 report passed its decision gate and matched the configured file,
split, and P1 hashes. It records the producer SpecID
`TAUSB-SDH-DGCAIP-CGR-E20-v2`. The R3 consumer uses
`TAUSB-SDH-DGCAIP-R3-DIAG-v1`. The runtime incorrectly requires these two
different provenance roles to have the same identifier.

## 3. Selected Repair

Add one explicit configuration field:

```yaml
dgcaip:
  expected_d0_spec_id: TAUSB-SDH-DGCAIP-CGR-E20-v2
```

The D0 runtime gate must compare `d0_report.spec_id` with
`dgcaip.expected_d0_spec_id`, not with the current consumer
`config.spec.spec_id`.

For legacy R2 configurations that do not define the new field, the runtime must
fall back to the existing consumer SpecID. This preserves current feature-off
behavior. R4 must require the explicit frozen field.

## 4. Checks That Must Remain Unchanged

- D0 file SHA256;
- D0 decision pass;
- D0 split hash;
- D0 source-P1-state SHA256;
- source P1 metrics SHA256;
- all dataset, secret, hiding, surrogate, seed, batch, step, epsilon, tolerance,
  routing, and backtracking bindings;
- default-off diagnostics behavior.

No objective, gradient route, constraint tolerance, learning rate, candidate,
backtracking schedule, or H1/H2 rule may change.

## 5. Required Local Tests

1. A frozen R4 config accepts the authentic D0 producer SpecID.
2. A wrong `expected_d0_spec_id` fails closed.
3. A D0 report whose SpecID differs from the explicit expected producer fails.
4. Legacy configs without the new field retain the previous comparison and
   behavior.
5. D0 hash, decision, split, and P1 mismatch branches still fail.
6. The full existing DGCAIP focused suite passes.
7. Controller/launcher Bash syntax, config hash binding, and stray-token scan
   pass.

## 6. Frozen R4 Diagnostic Protocol

R4 inherits R3 exactly:

- seed 0, VOC20, person id 14, image size 640;
- 16 calibration and 24 held-out samples, batch 4;
- eight mechanism steps and `eps=16/255`;
- P1-A, P1-B, P2-CAIP, and P4-DGCAIP only;
- no P3, EOT, JND, poisoned dataset, victim, AP50, E20, or E200;
- unchanged H1/H2 labels and thresholds;
- 600-second hard cap including initialization;
- automatic shutdown on success, failure, or hard cap.

The new ExpID and unique root are:

- ExpID: `TAUSB-SDH-DGCAIP-S0-R4-DIAG`
- artifact root:
  `/root/autodl-tmp/tausb-dgcaip-runs/TAUSB-SDH-DGCAIP-S0-R4-DIAG`

R3 artifacts and control paths must not be overwritten or deleted.

## 7. Success and Failure

Mechanical success requires the original R3 success signals: complete finite
traces, matching comparable input hashes, exact active-versus-recomputed
decisions, unchanged feature-off behavior, and exactly one H1 plus one H2 label.

Any Traceback, NaN, Inf, CUDA OOM, missing trace, decision mismatch, hard cap, or
missing controller terminal is a failure. All outcomes must be retained.

## 8. Claim Boundary

R4 may only report nonlinear rejection attribution and same-process replay. It
cannot claim improved unlearnability, non-target preservation, AP50,
robustness, transferability, or DG-CAIP effectiveness.

## 9. Execution Gate

After explicit approval: generate a new execution CSV, implement only the
selected binding repair, pass local and independent pre-run review, bind an
exact branch/commit/config hash, complete a no-card remote audit, and request
GPU enablement before the one permitted R4 run.
