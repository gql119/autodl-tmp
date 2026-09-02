# TAUSB-SDH-DGCAIP-RELAXED-PROMOTION-GATE-v4 review

Status: no-card gate passed; one guarded GPU v4 G1 is authorized.

## Execution identity

- Branch: `codex/tausb-sdh-dgcaip-relaxed-gate-v4`
- Execution commit: `72da79f936aaba1abc52158f1a52c9fc419b5c49`
- Remote checkout: `/root/autodl-tmp/_codex_worktrees/relaxed-v4-1d43c6f`
- Config SHA256: `73cc0617fa9721cef0f7b0b4ced6a107f4d271ab83e137410f7b3e81ea9e6331`

## Approved scope

This revision is limited to the v3 gate failure demonstrated by the preserved
G1 trace. It adds a separate `1e-6` nonlinear numeric comparison tolerance,
separates successful backtracking from actual skips, makes attack retention
diagnostic, and requires at least half of the eight updates plus final target
progress for promotion.

The target objective, carrier, component losses, risk bank, replay, route
geometry and primitive non-target tolerances remain unchanged. Existing user
dirty files and raw experiment artifacts are excluded from the revision.

## Validation evidence

- all changed Python modules and tests passed `py_compile`;
- the execution CSV parsed as five rows with a consistent 28-column schema;
- `git diff --check` passed;
- no user dirty file or experiment artifact is included in the v4 change set.

At the frozen execution commit, the four focused/regression files passed `66`
tests in `9.39s`. JUnit evidence:

- Path: `/root/autodl-tmp/tausb-dgcaip-preflight/relaxed-v4-72da79f/pytest.xml`
- SHA256: `ed94fb26a8c92f3d25bb1d0da460d87964b213f2d90d60763b8627a3e8723e09`

Remote `py_compile` and exact v4 config validation passed.

The initial formal preflight at the parent commit rejected the data disk because only
`5,095,329,792` bytes were free versus the legacy `5 GiB` threshold. The prior
same-shape v3 G1 used only `2.8 MiB` of artifacts and approximately `16 KiB` of
control/log data. The v4-only threshold is consequently registered at `4 GiB`;
legacy thresholds remain unchanged.

The revised exact-commit controller preflight passed:

- Schema/status: `tausb.dgcaip-g1-strict-preflight.v4` / `passed`
- GPU required: `false`
- Minimum/free bytes: `4,294,967,296` / `5,094,436,864`
- Risk coverage/replay slots: `1.0` / `32`
- Evidence: `/root/autodl-tmp/tausb-dgcaip-preflight/relaxed-v4-72da79f/preflight.json`
- SHA256: `d865fe10f4f0f37d2b66efc865ac721932389a6d095afa100d702abf6cd59610`

P1, G0 risk bank, replay manifest and e1/e5/e20 checkpoint hashes all matched
the frozen config. Artifact, control, log, cache and short temporary roots were
fresh. No GPU mechanism was run during review.

## GPU boundary

The next action is exactly one eight-step guarded v4 G1 using execution commit
`72da79f936aaba1abc52158f1a52c9fc419b5c49`, a 20-minute controller hard cap and
automatic shutdown on success or failure. Both passing and failed scientific
results must be retained. No G2, victim training, EOT or live parameter tuning
is authorized in the same boot.

## First GPU enable attempt

The first launch request on 2026-09-02 did not enter GPU computation. The
operator-supplied temporary root was
`/root/autodl-tmp/tausb-dgcaip-tmp/TAUSB-SDH-DGCAIP-S0-DSR-SCGR-V4-G1-R1`,
whose 71-byte resolved path exceeds the controller's 48-byte PyTorch AF_UNIX
socket limit. Preflight rejected the command before creating the control, log,
cache, temporary or formal artifact roots, and `--shutdown-on-exit` closed the
instance. This was a launch-parameter error rather than a method, loss or gate
result.

After no-card restart, the otherwise identical exact command passed preflight
with the reviewed short temporary root `/root/autodl-tmp/tv4g1r1`. The frozen
execution commit, config hash, artifact/control/log/cache roots and all
scientific parameters remain unchanged. The next action is one GPU retry using
that short temporary root; it is still the first actual v4 G1 computation.
