# TAUSB-SDH-DGCAIP-COMPONENT-ALIGNED-ROUTE-v3 review

Status: no-card gate passed; one guarded GPU G1 may be requested.

## Reviewed execution identity

- Branch: `codex/tausb-sdh-dgcaip-component-route-v3`
- Execution commit: `a316d748ff764c773208780aaa12dcfc3f2a69a6`
- Remote checkout: `/root/autodl-tmp/_codex_worktrees/component-v3-dev`
- Spec: `TAUSB-SDH-DGCAIP-COMPONENT-ALIGNED-ROUTE-v3`
- Config SHA256: `2139e1b0a23f3aa74dfce1f5bcb1ab08a8bf21e713550e824d5cbf7208018e72`

The later review/CSV-only commit is not part of the execution identity. GPU G1
must continue to use the exact detached execution commit above.

## Initial review

The v2 failure was reproduced conceptually and in a unit test: a combined
class loss can have a non-worsening gradient while an independently audited
component worsens. The v3 implementation replaces that mismatch with canonical
`snapshot/class/family` rows for NLA, probability, IoU, alignment and JS.

For every v3 row:

1. one tensor supplies the detached baseline and the gradient;
2. candidate evaluation reconstructs the same risk-weighted scalar;
3. gradient, baseline and candidate key registries must match exactly;
4. any missing or unexpected candidate key raises before acceptance;
5. v1/v2 helpers and schemas remain available and their route geometry is not
   duplicated.

The v3 final gate alone accepts post-cast target progress at
`0.60 - 1e-6`. The raw value is not altered, and v2 retains its exact historical
gate.

No blocking code-review finding remains after the strict candidate-key check
and the e1 checkpoint binding correction.

## No-card evidence

### Focused and regression tests

Remote command ran the following files with bytecode/cache writes disabled in
the checkout:

- `test_constraint_gradient_router.py`
- `test_dgcaip_strict_step.py`
- `test_dgcaip_experiment.py`
- `test_dgcaip_g1_strict.py`

Result: `59 passed in 6.80s`.

JUnit evidence:

- Remote path: `/root/autodl-tmp/tausb-dgcaip-preflight/component-v3-a316d74/pytest.xml`
- SHA256: `b7f0b18428f3ad169944be239a93f0a800d6daad0f07b8d02f54ea4b09324626`

The five modified Python modules also passed `py_compile` under Python 3.8.10.

### Formal controller preflight

The first preflight correctly rejected a 71-byte temporary path because the
existing AF_UNIX limit is 48 bytes. It created no experiment output. The final
preflight used `/root/autodl-tmp/t/v3g1-a316` and passed.

- Schema: `tausb.dgcaip-g1-strict-preflight.v3`
- Status: `passed`
- GPU required: `false`
- Dataset-risk coverage: `1.0`
- Replay slots: `32`
- Storage free bytes: `6402551808`
- Evidence path: `/root/autodl-tmp/tausb-dgcaip-preflight/component-v3-a316d74/preflight.json`
- Evidence SHA256: `70a0ec4c1bef7bc6d54b4ffab2ab5e2921ec4829844e06ae9b555ca918174a1a`

Verified bindings:

- P1: `2e102026a9356116de38acb1f5056bf5728afcd453e3447b516d4222f4d70b81`
- G0 canonical risk bank: `3dcc755fc7629cc5d2b37bd7b6931088001bf0ca0d3976343d7420d4236eb5fc`
- Replay manifest: `e5dd31cac06d038f4fc305970a9a60a2e2f34b3ae61e55af7405568fbbb7e457`
- e1: `6ebacf59d7fa27ae8d30bb86571d5f089392e19d52ba9ffd7fd204faa70c5ae1`
- e5: `cfaf454563e7ac81676468ec09fb08a94718a9902c5ee7057ee3db0d63202fc4`
- e20: `e660ed4b2f36e8b866f89a4f88a02e3d3a7eed6f2727f99573cc3c4d8bfaad53`

The artifact, control, log, cache and short temporary roots were all fresh at
preflight time. No GPU was visible and no mechanism step was run.

## GPU authorization boundary

The no-card gate authorizes only one eight-step guarded v3 G1 after the user
turns on GPU mode. Required controls remain:

- exact execution commit `a316d748ff764c773208780aaa12dcfc3f2a69a6`;
- 20-minute controller hard cap;
- automatic shutdown on success or failure;
- retain failed as well as passed artifacts;
- no live parameter tuning;
- no G2, M1, victim training, EOT or AP50 claim in the same boot.

Passing G1 would establish only local component-aligned routing and nonlinear
acceptance. It would not yet establish person unlearnability or non-target AP50
preservation.
