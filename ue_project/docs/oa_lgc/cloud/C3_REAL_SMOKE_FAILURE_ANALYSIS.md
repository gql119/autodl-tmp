# C3 Real-YOLO End-to-End Smoke Failure Analysis

Time: 2026-07-14

Branch: `codex/oa-lgc-real-yolo-pilot`

Starting commit: `8145a1a`

Environment: Windows 11; Python 3.12.13; PyTorch 2.11.0+cu128; Ultralytics 8.4.90; RTX 2070 8 GiB

Configuration: `configs/oa_lgc/cloud/c3_real_yolo_smoke.yaml`

Command: `python -m oa_lgc.real_yolo_smoke --config configs/oa_lgc/cloud/c3_real_yolo_smoke.yaml --output artifacts/oa_lgc/cloud/20260714_145816_C3_0`

Expected behavior: A-E complete with native loss, differentiable J=1/3/5 trajectories, delta updates, stable base hash, valid TAL/box/DFL diagnostics, at least one non-target gain per run, and same-seed reproduction.

Actual behavior: all 19 C3 Gate checks passed. Total matrix plus reproduction runtime was 27.73 seconds. Peak reported allocation was 589,785,088 bytes. No NaN, Inf, state mutation, overlap, low coverage, or proxy fallback was observed.

Traceback: none.

Cause: no failure was observed.

Fix: none required.

Result after fix: not applicable.

Historical impact: none. Historical files and artifacts were not overwritten or deleted.

Next stage: allowed. No blocking failure was triggered.
