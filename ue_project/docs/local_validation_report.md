# Local P1/P2 Validation Report

## Git And Environment

- HEAD: `aacab4d2b42cb27c237c87e39dcfcfb363c58807`
- Branch: `codex/det-crad-p1-p2`
- Python: `F:\autodl-tmp\ue_project\.venv\Scripts\python.exe`
- Torch: `2.11.0+cu128`, CUDA: `12.8`
- GPU: `NVIDIA GeForce RTX 2070`

## Architecture Check

- New-method forbidden import hits: `{'alce': [], 'rlcp': [], 'context prototype': [], 'des-r': [], 'fdacb': [], 'weighted ring': [], 'tausb_universal': []}`
- Functional scores max abs diff: `0.09679985046386719`
- Functional parameter leak max abs diff: `0.0`

## Category Loss Isolation

```json
{
  "ambiguous_positive_count": 0,
  "authorized_class_ids": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    16,
    17,
    18,
    19
  ],
  "authorized_loss": 15.977442741394043,
  "authorized_loss_grad_on_authorized_assigned_class_max": 0.1446397751569748,
  "authorized_loss_grad_on_protected_assigned_class_max": 0.0,
  "authorized_positive_count": 20,
  "protected_class_id": 14,
  "protected_loss": 13.18771743774414,
  "protected_loss_grad_on_authorized_assigned_class_max": 0.0,
  "protected_loss_grad_on_protected_assigned_class_max": 0.0901946946978569,
  "protected_positive_count": 20
}
```

## P1 Gradient Validation

- cos_protected_clean_poison: `0.986454963684082`
- cos_authorized_clean_poison: `0.9785535931587219`
- p1_loss: `1.0079014301300049`
- delta_grad_norm: `0.3872668445110321`
- surrogate_parameter_max_abs_diff: `0.0`
- 20-step sequence saved in `F:\autodl-tmp\ue_project\outputs\local_validation_p1_20_final\p1_20_step_sequence.json`

## P2 Virtual Update Validation

- virtual_parameter_update_norm: `0.0206398144364357`
- parameter_leak_max_abs_diff: `0.0`
- query loss before update: `11.733359336853027`
- query loss after clean update: `11.72040605545044`
- query loss after poisoned update: `11.720501899719238`
- protected_learning_gap: `0.00025463104248046875`
- authorized_learning_gap: `0.00015878677368164062`
- meta_gradient_norm_to_delta: `0.04824462905526161`

## Memory Stability

- iterations: `20`
- allocated first/last/max MB: `108.93603515625` / `108.93603515625` / `108.93603515625`
- allocated slope MB/iter: `-1.3347697174475514e-15`

## Legacy-Best Compatibility

```json
{
  "is_poisoned": true,
  "linf": 0.0627450942993164,
  "losses": {
    "L_budget": 0.8758174777030945,
    "L_total": 0.8758174777030945
  },
  "note": "poisoned",
  "ok": true,
  "perturbed_area_ratio": 0.024053333333333333,
  "sample_id": "000066",
  "support_ratio": 0.017498666666666666,
  "support_source": "forced_pseudo_fallback"
}
```

## Unit Tests

- tests/run_tests.py returncode: `0`
- pytest returncode: `0`

## Unresolved Issues

- None from this local validation run.

## Recommendation

Mechanism checks pass locally; it is reasonable to proceed to a small victim retraining smoke.
