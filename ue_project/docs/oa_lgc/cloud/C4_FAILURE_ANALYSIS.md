# C4 Failure Analysis

Time: 2026-07-14

Branch: `codex/oa-lgc-real-yolo-pilot`

Starting commit: `7c0d8ba`

Environment: local Windows workspace plus the existing read-only VOC/checkpoint data under `F:/autodl-tmp/ue_project`

Configuration: `configs/oa_lgc/cloud/c4_protocol_audit.yaml`

Command: `python -m oa_lgc.protocol_audit --config configs/oa_lgc/cloud/c4_protocol_audit.yaml --output artifacts/oa_lgc/cloud/20260714_151949_C4_0`

Expected behavior: recover VOC2007 trainval + VOC2012 trainval as training, VOC2007 test as independent test, prove zero overlap, train or verify the corresponding clean baseline, and derive `E_pilot` from per-epoch target/non-target curves.

Actual behavior: only VOC2007 trainval is present. VOC2012 trainval and VOC2007 test are absent. The historical mini baseline is hash-verifiable but protocol-ineligible and lacks the required per-epoch class-group curves.

Traceback: none from the definitive audit. An earlier broad read-only filesystem search timed out after entering large historical output trees; it was safely terminated and replaced with targeted searches.

Cause: required local data components and matching clean-baseline evidence are not available in the authorized workspace.

Fix: no in-scope automatic fix is allowed. The user must provide local VOC2012 trainval and VOC2007 test data with authoritative split files, or explicitly authorize a different auditable protocol. Silent download and split guessing are forbidden.

Result after fix: not applicable; the blocking prerequisites remain absent.

Historical impact: none. No dataset, checkpoint, historical artifact, or dirty file was changed or deleted.

Next stage: C5 is not allowed. C4 is `blocked`.
