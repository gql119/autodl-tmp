# Supervision Decomposition P0 Report

## Git And Code

- base commit: `e02a7528e9a074d229ed02383ecb7191425595dd`
- diagnostic-run HEAD: `fe68d1eaf2b1a3da26d428dab86de54b1bf3d5b8`
- branch: `codex/supervision-decomposition-p0`
- working-tree status: `?? AGENTS.md
?? p0_pre_branch_untracked.txt
?? p0_pre_branch_worktree.patch
?? p1_p2_before_local_validation.patch
?? p1_p2_untracked_before_local_validation.txt
?? scripts/
?? ue_framework/datasets/
?? ue_framework/tools/
?? ue_framework/utils/robust_transforms.py
?? "\345\210\235\347\250\277/"`
- key file hash: `{'localized_support': '7b2047bcdea7a01b30be1cb2cae3c9c06de04f459523bd8e164e6a9636633b4f', 'supervision_decomposer': '5fd5b923c53d7c9de08abffdf558eabb65fb27dcbae3ae3ef0092f4789b2be69', 'runner': '187eb5270765839e50c4e0dfe8e7df0083a1d444b3a991663c567cb7e328ad46', 'config': '7af003608a900433a9c29037adb83b03eee7ad70f4f59d2c425a0cf15de7de34'}`

## Localized Support

- max valid support ratio: `0.7339843511581421`
- max outside-support delta: `0.0`
- support rows: `outputs\supervision_decomposition_p0\support_statistics.csv`
- still full-image mask: `False`

## Supervision Statistics

- batch rows: `outputs\supervision_decomposition_p0\batch_results.csv`
- selected image ids: `{'person_only': ['000066', '000110', '000113', '000138', '000146'], 'authorized_only': ['000005', '000007', '000012', '000016', '000019'], 'low_overlap': ['000032', '000041', '000050', '000051', '000083'], 'high_overlap': ['000021', '000023', '000048', '000129', '000173']}`

## Loss Reconstruction

- max relative reconstruction error: `8.22462595806428e-08`
- max decomposer-vs-Ultralytics full loss delta: `9.5367431640625e-07`
- reconstruction rows: `outputs\supervision_decomposition_p0\loss_reconstruction.csv`

## Interventions

- person_assigned_class: protected_delta=0.012992650270462036, authorized_delta=0.0, shared_delta=9.5367431640625e-07, ok=True
- authorized_assigned_class: protected_delta=0.0, authorized_delta=0.0899156928062439, shared_delta=9.5367431640625e-07, ok=True
- person_other_class: protected_delta=0.0, authorized_delta=0.0, shared_delta=0.5514516830444336, ok=True
- authorized_person_negative: protected_delta=0.0, authorized_delta=0.0, shared_delta=0.5514516830444336, ok=True
- background_logits: protected_delta=0.0, authorized_delta=0.0, shared_delta=11.029082298278809, ok=True

## Box/DFL Isolation

- person_box_dfl: protected_delta=5.849999904632568, authorized_delta=0.0, shared_delta=0.0, ok=True
- authorized_box_dfl: protected_delta=0.0, authorized_delta=5.100000381469727, shared_delta=0.0, ok=True
- ambiguous_box_dfl: protected_delta=0.0, authorized_delta=0.0, shared_delta=5.849999904632568, ok=True

## Gradient Leakage Matrix

- matrices: `outputs\supervision_decomposition_p0\gradient_leakage_matrices.json`

## Validation

- `python tests/run_tests.py`: `ALL_PASS`
- `pytest tests -q`: `26 passed in 5.23s`
- legacy-best smoke: `ok=True`, support_source=`forced_pseudo_fallback`, linf=`0.0627450942993164`
- victim retraining: not run
- poisoned dataset generation: not run

## Conclusion

1. localized support fixed: `True`
2. outside support delta strictly zero: `True`
3. protected assigned-class loss isolated: `true`
4. authorized assigned-class loss isolated: `true`
5. non-assigned-class logits route to shared: `true`
6. background negatives route to shared: `true`
7. ambiguous units route to shared: `True`
8. box/DFL split by assigned GT class: `true`
9. losses reconstruct original full loss: `True`
10. recommend entering J=3 learning-gain stage: `True`
