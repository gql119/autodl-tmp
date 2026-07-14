# OA-LGC Real-YOLO Final Report

## 1. Overall status

`real YOLO engineering pass, pilot blocked`

The real-detector functional engineering chain passed C0-C3. C4 could not establish the required VOC2007+2012 training / VOC2007 test protocol, so C5 was not started. No method-effectiveness claim is made.

## 2. Git status

- Branch: `codex/oa-lgc-real-yolo-pilot`.
- Start commit: `04448a338239863d71a12198ede2fb08980be3a0`.
- C0: `c713756`.
- C1: `7cd65bc`.
- C2: `8145a1a`.
- C3: `7c0d8ba`.
- C4: `de0ae5d`.
- Push: all stage commits pushed to `origin/codex/oa-lgc-real-yolo-pilot`.
- Historical artifacts overwritten: no.
- Six pre-existing dirty tracked `ue_framework` files remain untouched and uncommitted.

## 3. Stage table

| Stage | Status | Main result | Evidence | Commit |
| --- | --- | --- | --- | --- |
| C0 | pass | CUDA, mini VOC, native YOLO forward/loss/TAL available | `artifacts/oa_lgc/cloud/20260714_141729_C0_0/` | `c713756` |
| C1 | pass | Functional adapter and mixed derivative pass | `artifacts/oa_lgc/cloud/20260714_143121_C1_0/` | `7cd65bc` |
| C2 | pass | TAL coverage and per-class box/DFL diagnostics pass | `artifacts/oa_lgc/cloud/20260714_144529_C2_0/` | `8145a1a` |
| C3 | pass | A-E real-detector engineering matrix and reproduction pass | `artifacts/oa_lgc/cloud/20260714_145816_C3_0/` | `7c0d8ba` |
| C4 | blocked | Missing VOC2012 trainval and VOC2007 test; no legal E_pilot | `artifacts/oa_lgc/cloud/20260714_151949_C4_0/` | `de0ae5d` |
| C5 | blocked | Not started because C4 Gate failed | `docs/oa_lgc/cloud/C4_FAILURE_ANALYSIS.md` | none |
| C6 | pass | Decision and recoverable handoff completed | this report | final commit |

## 4. Real YOLO engineering

- Adapter: `YOLOFunctionalAdapter`, backend `real_ultralytics_yolo`.
- Functional backend: `torch.func.functional_call`.
- Native inner loss: box + classification + DFL with real TAL.
- classification-head parameters: 24 tensors / 373,308 parameters.
- detection-head parameters: 48 tensors / 755,196 parameters.
- selected neck/head parameters: 84 tensors / 1,409,148 parameters.
- Buffer handling: independent cloned buffers for clean and poison trajectories.
- J=1: classification and detection head pass.
- J=3: classification and detection head pass.
- J=5: classification head single-episode pass; optional detection-head run not required.
- Protect-only mixed derivative: non-zero; C1 norm 12.078218.
- Base model mutation: none; hash remained `25c0ad56...eec28c`.
- Reproducibility: Run A gain and final-delta max differences were both 0.

## 5. TAL / Box / DFL

- C2 target coverage median/minimum: 1.0/1.0.
- Low-coverage episode ratio: 0.0.
- Target box loss available; median 1.544335.
- Target DFL loss available; median 1.228376.
- Valid non-target classes: bicycle, dog, horse.
- Target assignment overlap median: 1.0.
- Diagnostic schema: complete target and 20-class rows with explicit invalid reasons.

## 6. Pilot

- Data protocol: blocked. Only VOC2007 trainval is local; VOC2012 trainval and VOC2007 test are absent.
- Clean baseline: formal C4 baseline not run.
- Historical baseline: recoverable but protocol-ineligible 800/200 VOC2007-trainval mini split.
- E_pilot: not determined; historical curve lacks per-epoch person and non-target AP.
- Settings completed: none of P0-P8.
- Learning-gain/AP trend, correlation, Pareto: unavailable because C5 was not started.

## 7. Failures

1. C0 Unicode workspace path was corrupted across the external interpreter boundary. Fixed by using the SHA-identical ASCII-path data/checkpoint while loading current branch code through `PYTHONPATH`.
2. Legacy checkpoint runtime args lacked attribute-style box/cls/DFL fields. Fixed with an explicit Ultralytics configuration merge.
3. EMA checkpoint parameters loaded frozen. Fixed by explicit fast-parameter gradient setup.
4. Initial reproducibility test demanded bitwise CUDA equality. Corrected to the preregistered `1e-7` tolerance; authoritative run was exact.
5. CUDA peak-memory reset rejected device arguments on this Windows build. Replaced by non-resetting peak queries.
6. C4 blocking failure: VOC2012 trainval, VOC2007 test, and eligible per-epoch clean curves are absent. No automatic in-scope fix is permitted.

## 8. Test results

- Existing tests: 91 passed before this branch.
- New adapter tests: 16 passed.
- New diagnostics tests: 10 passed.
- Final combined result: 117 passed in 10.50 seconds.
- Failed: 0.
- Skipped: 0.

## 9. Reproduction commands

```powershell
$env:PYTHONPATH=(Get-Location).Path
$python='F:\autodl-tmp\ue_project\.venv\Scripts\python.exe'

# C1 (use a new unique output path)
& $python -m oa_lgc.yolo_adapter_smoke --config configs/oa_lgc/cloud/c1_yolo_adapter.yaml --output artifacts/oa_lgc/cloud/<new_run_id>_C1_0

# C2
& $python -m oa_lgc.yolo_diagnostics_smoke --config configs/oa_lgc/cloud/c2_yolo_diagnostics.yaml --output artifacts/oa_lgc/cloud/<new_run_id>_C2_0

# C3
& $python -m oa_lgc.real_yolo_smoke --config configs/oa_lgc/cloud/c3_real_yolo_smoke.yaml --output artifacts/oa_lgc/cloud/<new_run_id>_C3_0

# C4 audit only; expected status is blocked until local data are supplied
& $python -m oa_lgc.protocol_audit --config configs/oa_lgc/cloud/c4_protocol_audit.yaml --output artifacts/oa_lgc/cloud/<new_run_id>_C4_0

# C5: intentionally no command; C4 Gate is blocked.
```

## 10. Documents and artifacts

Progress and final decision:

- `docs/oa_lgc/cloud/OA_LGC_CLOUD_PROGRESS.md`
- `docs/oa_lgc/cloud/OA_LGC_REAL_YOLO_FINAL_REPORT.md`
- `docs/oa_lgc/cloud/OA_LGC_NEXT_EXPERIMENT_DECISION.md`

Stage documents:

- C0: `C0_PREFLIGHT_AUDIT.md`, `C0_ENVIRONMENT.md`, `C0_IMPLEMENTATION_PLAN.md`, `C0_FAILURE_ANALYSIS.md`.
- C1: `C1_YOLO_ADAPTER_PLAN.md`, `C1_YOLO_ADAPTER_IMPLEMENTATION.md`, `C1_YOLO_ADAPTER_REPORT.md`, `C1_YOLO_ADAPTER_FAILURE_ANALYSIS.md`.
- C2: `C2_TAL_DFL_PLAN.md`, `C2_TAL_DFL_IMPLEMENTATION.md`, `C2_TAL_DFL_REPORT.md`, `C2_TAL_DFL_FAILURE_ANALYSIS.md`.
- C3: `C3_REAL_SMOKE_PLAN.md`, `C3_REAL_SMOKE_REPORT.md`, `C3_REAL_SMOKE_FAILURE_ANALYSIS.md`.
- C4: `C4_PROTOCOL_PLAN.md`, `C4_PROTOCOL_AUDIT.md`, `C4_CLEAN_BASELINE_REPORT.md`, `C4_FAILURE_ANALYSIS.md`.

Authoritative artifacts are the C0-C4 paths in the stage table. Failed or superseded unique runs were preserved and never reused.

## 11. Decision

Do not enter formal multi-seed, multi-class, or multi-model experiments yet.

The real-YOLO engineering basis is ready, but the pilot is blocked before any AP/correlation/Pareto result exists. Formal expansion requires an auditable local VOC protocol, a converged clean baseline with per-epoch target/non-target curves, a valid `E_pilot`, and completion of the preregistered P0-P7 matrix.
