# J3 Learning Gain V1 Report

## Git And Implementation

- base commit: `bb4a1a7d651ec618394479ff0a5d84d2fe8295ce`
- diagnostic HEAD: `bb4a1a7d651ec618394479ff0a5d84d2fe8295ce`
- branch: `codex/j3-learning-gain-v1`
- tracked changes present during run: `2`
- key file hash: `{'functional_optimizer': 'cdfab60f43554b0b7273aea2c5fd802bea3f84c62cb2fd3c07555e5f53e494db', 'learning_gain': '362e87bf544b8fb7e6b2bdd32042307491c911920ab590e47f1b276fdaac68da', 'rollout_engine': '97274a6de076edc1373d62ba7fbab8f057789ccc9959f5c015a748b5bc5c8d69', 'runner': '2c7134ebf57387b6862d7ec8b046ac23a6ded1e02fd308619c70312958212fc7'}`
- resolved config: `outputs\j3_learning_gain_v1\config_resolved.yaml`
- optimization curve: `outputs\j3_learning_gain_v1\optimization_curve.csv`

## Rollout Correctness

- J=3 executed: `true`
- clean/poison initial parameters identical: `true`
- batch sequence matched: `true`
- augmentation matched: `resize-only, same seed recorded`
- optimizer state identical but independent: `true`
- dynamic TAL each step: `true`
- finite gradient to delta: `2.3767874240875244`
- finite-difference loss before/after: `0.09972193837165833` / `0.09876071661710739`
- finite-difference ok: `True`
- surrogate parameter leak max: `0.0`

## Learning Gain

- start held-out D_t/E_a/E_s/S: `0.000997462598494773` / `0.00026747300435090435` / `0.003596485285864522` / `-0.0028664957533086027`
- end held-out D_t/E_a/E_s/S: `-3.1859269142052806` / `0.8148860211173693` / `0.6196052461862565` / `-4.6204183280467985`
- end train D_t/E_a/E_s/S: `0.11038048374881934` / `0.8928988139067466` / `0.6033608010659616` / `-1.38587914891541`

## Invalid Trajectories

- start held-out protected valid ratio: `0.5`
- end held-out protected valid ratio: `0.5`
- end held-out authorized valid ratio: `0.6666666666666666`

## J1 vs J3

- J3 held-out S_gain: `-4.6204183280467985`
- J1 held-out S_gain: `-2.425524580758065`
- J3 better than J1: `False`

## Memory

- memory profile rows: `outputs\j3_learning_gain_v1\memory_profile.csv`
- allocated first/last: `136245760` / `136244736`

## Conclusion

- `B. 代理只能在训练轨迹生效，存在 trajectory overfitting`