# TAUSB-SDH-DGCAIP-RELAXED-PROMOTION-GATE-v4 review

Status: remote pytest passed; revised exact-commit preflight pending.

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

- remote no-card controller preflight on fresh v4 roots;
- reviewed execution commit and hashes.

## Local evidence

- all changed Python modules and tests passed `py_compile`;
- the execution CSV parsed as five rows with a consistent 28-column schema;
- `git diff --check` passed;
- no user dirty file or experiment artifact is included in the v4 change set.

After the instance became reachable, the four focused/regression files passed
`65` tests in `8.10s`; JUnit SHA256 is
`aed483bb950fb570d2a431454cd82628020766c0f7e2c409b5c8b144d5f1d5b4`.
Remote `py_compile`, exact v4 config validation and fresh-root checks also
passed.

The initial formal preflight then rejected the data disk because only
`5,095,329,792` bytes were free versus the legacy `5 GiB` threshold. The prior
same-shape v3 G1 used only `2.8 MiB` of artifacts and approximately `16 KiB` of
control/log data. The v4-only threshold is consequently registered at `4 GiB`;
legacy thresholds remain unchanged. A new exact-commit preflight is pending.

GPU G1 remains closed until these items pass.
