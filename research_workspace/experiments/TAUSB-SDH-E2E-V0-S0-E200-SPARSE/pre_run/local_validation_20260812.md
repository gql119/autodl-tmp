# TAUSB-SDH-E2E-V0-SPARSE-E200-v1 local validation

Date: 2026-08-12
Scope: mechanical/no-card validation only; this is not E200 experimental evidence.

## Result

- Focused and regression tests: `106 passed in 14.03s`.
- Python 3.8 AST gate: 5 changed runtime modules passed.
- Git Bash syntax gate: `sparse_e200_controller_data_disk_v1.sh` passed `bash -n`.
- Controller CLI exposes `--victim-epochs {20,200}`, `--cache-root`, and `--tmp-root`; default victim horizon remains 20.
- Binder CLI preserves legacy `--e20-only` and exposes `--full-voc-only --victim-epochs {20,200}`.
- Real YOLOv8n config construction and canonical init tensor hashing passed:
  - hash: `3376bfad314962c3ae94bb9f95b4395bcf4b33bb275255e84090f6591afed512`
  - length: 64 hexadecimal characters.
- Real local VOC dataloader probe passed on `000005.jpg`:
  - dataset count: 1;
  - batch shape: `[1, 3, 480, 640]`;
  - batch label count: 3;
  - image SHA256: `fbb7fad63f242bf15fef88a2532b362241586cba4bf9dd0f26c09b94e42c5659`;
  - label SHA256: `9d68a888d5b0b628888e0266768cfdb07594fcff593883803954043a2917e42f`;
  - train-list SHA256: `96e59ac444d086653587df889c0188c79e525aabe5a2c10477548390179c52ac`.
- `git diff --check` passed for the scoped E200 files.

## Covered invariants

- E20 binder/config/controller compatibility remains tested.
- E200 binds C0/M1 to 200 epochs and distinct E200 roots/IDs.
- Overall E200 wall cap is 32,400 seconds; each arm train+evaluate cap is 12,600 seconds.
- E200 Success/Failure/Inconclusive thresholds are distinct from E20 feasibility thresholds.
- C0 sanity blocks M1 when person AP50 is below 0.60 or non-target macro AP50 is below 0.50.
- Fresh victim init evidence is written before training; M1 refuses to call `model.train` if its init hash differs from C0.
- Cache/tmp/output roots must bind to the mounted data disk for E200.
- Success, scientific failure, inconclusive, operational failure, and timeout retain a SHA256 terminal evidence manifest.
- Wrapper installs the shutdown trap before validating required inputs and flushes evidence before shutdown.

## Validation gaps reserved for remote pre-run review

- Windows cannot validate the AutoDL Linux device IDs, `/root/autodl-tmp` mountpoint, live free space, or system-disk stage growth. These remain fail-closed controller prechecks.
- No local GPU training, E200 epoch, AP50 evaluation, or shutdown command was executed.
- Actual frozen remote mechanism/P1 paths and surrogate checkpoint are re-hashed on the exact remote checkout before any training.
- Wrapper behavior was syntax-checked and structurally reviewed; actual AutoDL shutdown is intentionally deferred to the authorized remote run.

## Claim boundary

This report establishes code-path and input-binding readiness only. It does not establish that the method succeeds, fails, or preserves any AP50 metric at E200.
