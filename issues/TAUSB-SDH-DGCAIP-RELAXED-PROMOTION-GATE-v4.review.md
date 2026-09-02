# TAUSB-SDH-DGCAIP-RELAXED-PROMOTION-GATE-v4 review

Status: local implementation complete; remote no-card pytest and preflight pending.

## Approved scope

This revision is limited to the v3 gate failure demonstrated by the preserved
G1 trace. It adds a separate `1e-6` nonlinear numeric comparison tolerance,
separates successful backtracking from actual skips, makes attack retention
diagnostic, and requires at least half of the eight updates plus final target
progress for promotion.

The target objective, carrier, component losses, risk bank, replay, route
geometry and primitive non-target tolerances remain unchanged. Existing user
dirty files and raw experiment artifacts are excluded from the revision.

## Pending evidence

- focused and regression pytest result (the bundled local Python has no
  PyTorch/pytest, so this must run in the existing AutoDL environment);
- exact config validation;
- remote no-card controller preflight on fresh v4 roots;
- reviewed execution commit and hashes.

## Local evidence

- all changed Python modules and tests passed `py_compile`;
- the execution CSV parsed as five rows with a consistent 28-column schema;
- `git diff --check` passed;
- no user dirty file or experiment artifact is included in the v4 change set.

The configured SSH endpoint returned connection refused during this review, so
remote results are deliberately not claimed.

GPU G1 remains closed until these items pass.
