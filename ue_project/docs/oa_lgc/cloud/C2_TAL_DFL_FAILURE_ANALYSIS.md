# C2 TAL / Box / DFL Failure Analysis

Time: 2026-07-14

Branch: `codex/oa-lgc-real-yolo-pilot`

Starting commit: `7cd65bc`

Environment: Windows 11; Python 3.12.13; PyTorch 2.11.0+cu128; Ultralytics 8.4.90; RTX 2070 8 GiB

Configuration: `configs/oa_lgc/cloud/c2_yolo_diagnostics.yaml`

Command: `python -m oa_lgc.yolo_diagnostics_smoke --config configs/oa_lgc/cloud/c2_yolo_diagnostics.yaml --output artifacts/oa_lgc/cloud/20260714_144529_C2_0`

Expected behavior: all three episodes retain at least 50% target coverage; target box and DFL remain finite; at least one non-target class is valid; schemas are complete.

Actual behavior: all three episodes had target coverage 1.0, no low-coverage episode, finite positive box/DFL losses, and three valid non-target classes. The complete regression suite reported 117 passed.

Traceback: none.

Cause: no failure was observed.

Fix: none required.

Result after fix: not applicable.

Historical impact: none; protected TAL, training, evaluation, and historical artifact files were not modified.

Next stage: allowed. No blocking failure was triggered.
