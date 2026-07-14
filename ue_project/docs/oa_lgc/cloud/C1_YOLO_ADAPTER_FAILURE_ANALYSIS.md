# C1 Real-YOLO Adapter Failure Analysis

Time: 2026-07-14

Branch: `codex/oa-lgc-real-yolo-pilot`

Starting commit: `c713756`

Environment: Windows 11; Python 3.12.13; PyTorch 2.11.0+cu128; Ultralytics 8.4.90; RTX 2070 8 GiB

Configuration: `configs/oa_lgc/cloud/c1_yolo_adapter.yaml`

## Failure 1: bitwise CUDA reproducibility assertion

- Command: `python -m pytest tests/test_oa_lgc_yolo_adapter.py -q`.
- Expected: repeated J=1 fast states satisfy the documented tolerance.
- Actual: 15 tests passed and `test_real_yolo_reproducibility` failed because it required `torch.equal` bitwise identity.
- Traceback: assertion failed at the generator comparing every selected tensor with `torch.equal`.
- Cause: the test was stricter than the protocol, which asks for tolerance-bounded consistency. Step losses and parameter delta norms already matched exactly.
- Fix: compute and assert the maximum selected-parameter absolute difference is at most `1e-7`.
- Result after fix: `16 passed in 7.21s`; authoritative smoke measured maximum difference `0.0`.
- Historical impact: none.
- Next stage: allowed.

## Failure 2: Windows CUDA peak-memory reset argument

- Command: C1 smoke with a unique output directory.
- Expected: reset peak memory counters before the smoke.
- Actual: `RuntimeError: Invalid device argument` from `torch._C._cuda_resetPeakMemoryStats` for both `torch.device('cuda:0')` and integer 0.
- Cause: this Windows/PyTorch build does not accept the reset call used by the first implementation.
- Fix: do not reset the global counter; read the process peak with the no-argument query and record it as an absolute peak.
- Result after fix: smoke passed and wrote all required artifacts.
- Historical impact: none. The two failed unique directories are preserved and were not reused.
- Next stage: allowed.

No blocking failure was triggered. C1 Gate passed, so C2 is allowed.
