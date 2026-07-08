# Local P1/P2 Validation Report

## Git And Environment

- HEAD: `e616b4ee10201d621266b7ac0bd756b938aabee8`
- Branch: `codex/det-crad-p1-p2`
- Python: `F:\autodl-tmp\ue_project\.venv\Scripts\python.exe`
- Torch: `2.11.0+cu128`, CUDA: `12.8`
- GPU: `NVIDIA GeForce RTX 2070`

## Architecture Check

- New-method forbidden import hits: `{'alce': [], 'rlcp': [], 'context prototype': [], 'des-r': [], 'fdacb': [], 'weighted ring': [], 'tausb_universal': []}`
- Functional scores max abs diff: `0.12082958221435547`
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
  "authorized_loss": 14.020730972290039,
  "authorized_loss_grad_on_authorized_assigned_class_max": 0.06258975714445114,
  "authorized_loss_grad_on_protected_assigned_class_max": 0.0,
  "authorized_positive_count": 20,
  "protected_class_id": 14,
  "protected_loss": 15.921144485473633,
  "protected_loss_grad_on_authorized_assigned_class_max": 0.0,
  "protected_loss_grad_on_protected_assigned_class_max": 0.07880455255508423,
  "protected_positive_count": 20
}
```

## P1 Gradient Validation

- cos_protected_clean_poison: `0.9845537543296814`
- cos_authorized_clean_poison: `0.7797325253486633`
- p1_loss: `1.204821228981018`
- delta_grad_norm: `4.610808372497559`
- surrogate_parameter_max_abs_diff: `0.0`
- 20-step sequence saved in `F:\autodl-tmp\ue_project\outputs\local_validation\p1_20_step_sequence.json`

## P2 Virtual Update Validation

- virtual_parameter_update_norm: `0.013594443909823895`
- parameter_leak_max_abs_diff: `0.0`
- query loss before update: `8.274872779846191`
- query loss after clean update: `8.275852680206299`
- query loss after poisoned update: `8.276276588439941`
- protected_learning_gap: `0.0005359649658203125`
- authorized_learning_gap: `0.00011205673217773438`
- meta_gradient_norm_to_delta: `0.02276652306318283`

## Memory Stability

- iterations: `50`
- allocated first/last/max MB: `139.34033203125` / `139.34033203125` / `139.34033203125`
- allocated slope MB/iter: `-3.816382384693336e-16`

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
