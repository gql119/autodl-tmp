# Mini CS-DEM Stage 0 Preflight Audit

## Repository and environment

- Branch: `codex/mini-csdem-3stage`
- Starting commit: `a49e7b823e14c5290f085ac978499cd5ba733177`
- Remote: `origin https://github.com/gql119/autodl-tmp.git`
- Pre-existing worktree state: no tracked modifications; multiple user-owned untracked files/directories were present and were not modified or staged.
- Python: `F:\autodl-tmp\ue_project\.venv\Scripts\python.exe`, Python 3.12.13
- PyTorch / CUDA: PyTorch 2.11.0+cu128, CUDA available
- GPU: NVIDIA GeForce RTX 2070, 8192 MiB
- Ultralytics: 8.4.90

## Project and entry points

The historical experiment entry is `ue_framework/launch_one.py`. Its stages are implemented in `ue_framework/stages/generate.py`, `train_victim.py`, and `evaluate.py`; TAUSB is implemented in `ue_framework/methods/tausb_universal.py`.

The mini CS-DEM experiment is intentionally isolated behind `runners/run_mini_csdem.py` and `configs/mini_csdem/clean_baseline.yaml`. Stage 0 does not call the historical method factory or poison-generation stage.

Model initialization is `configs/voc_yolov8n_20cls.yaml`, a YOLOv8n-style 20-class detector (3,014,748 parameters). It is used instead of the upstream COCO `yolov8n.yaml` because it preserves the project VOC20 class space. Construction is from YAML, not a `.pt` file; `pretrained=false` and `resume=false` are passed explicitly.

Dataset source is the local VOC2007 `trainval` set at `outputs/local_validation/voc_raw/VOCdevkit/VOC2007`. The class names are read from the experiment data configuration. Resolving the unique name `person` gives class index `14`; the runner does not take a hard-coded target ID.

## Legacy-module audit

All “enabled” values below refer to the Stage 0 clean configuration and its independent CLI.

| Legacy component | Definition / configuration entry | Enabled | Training graph or image/loss effect | Hook registration |
|---|---|---:|---|---:|
| ALCE / context prototype / RLCP / PAG | `ue_framework/methods/alce_acgt.py`, `alce_losses.py`, and TAUSB ALCE config | No | Not imported; no input or loss effect | No |
| Shortcut prototype / old Fourier universal perturbation | `ue_framework/methods/fourier.py`, `ours.py`, `tausb_universal.py` | No | Not imported; no input effect | No |
| DES-R / FDACB / weighted ring / dual carrier | historical TAUSB configs and branches in `tausb_universal.py` | No | Not constructed; no graph effect | No |
| Feature preservation / late repair | TAUSB preservation losses and `enable_late_nt_repair` path | No | Not constructed; no loss effect | No |
| Pseudo / forced fallback | `allow_pseudo_mask_fallback` and `force_pseudo_mask_fallback` in legacy data/method configs | No | Legacy configs are not loaded | No |
| Old mid-feature / FPN hooks | `methods/base.py`, `core/feature_hooks.py`, and TAUSB activation hooks | No | Adapter is not instantiated during victim training | No; measured hook count before training = 0 |
| MTEPI channel hook | `ue_framework/methods/mtepi/stage2.py` | No | Independent method not imported | No |
| Old loss hook/callback | historical method and stage code | No | Ultralytics standard detection loss only | No custom callback |
| Automatic checkpoint load | historical evaluator/victim resume paths | No | model source is YAML; no `.pt` source | N/A |
| Automatic resume | legacy default in `ue_framework/config.py`; launch overrides for fresh runs | No | mini config and train call both set `resume=false` | N/A |
| Pretrained weights | possible in generic Ultralytics flows | No | train call explicitly records `pretrained=false` | N/A |

No historical code was deleted or changed. Isolation through a new CLI and feature flags preserves every legacy reproduction path. The clean config additionally sets all future mini CS-DEM feature flags to false.

## Deterministic mini split

- Manifest: `data_splits/mini_csdem_voc_seed0.json`
- File SHA256: `bc1ffad0d4a6167bb6882e1d137ab615dbf64e4f9c7589361bfb7518332216cd`
- Train: 800 images, 400 person-present, 400 person-free, 326 person/non-target cooccurrence images
- Validation: 200 images, 100 person-present, 100 person-free, 81 cooccurrence images
- Every VOC class has validation instances; the minimum is 8 instances.
- Images and labels are materialized only below ignored `outputs/mini_csdem/`; dataset files and caches are not committed.

## One-epoch smoke test

Command:

```powershell
.\.venv\Scripts\python.exe runners\run_mini_csdem.py --mode smoke --config configs\mini_csdem\clean_baseline.yaml
```

Results:

- PASS: model initialized from YAML with `pretrained=false`, `resume=false`.
- PASS: legacy hook count before training was 0.
- PASS: sampled clean image SHA256 was unchanged after training.
- PASS: training values were finite; no NaN/Inf.
- PASS: train and validation entry points completed.
- Validation mAP50 after one epoch was 0.0, which is expected for an early random-initialized detector and is not a mechanical failure.
- Machine-readable record: `outputs/mini_csdem/smoke_test.json` (ignored runtime artifact).

## Risks

- The mini split is much smaller than the full VOC protocol, so absolute AP is low and seed variance may be large.
- The RTX 2070 requires batch 16 and workers 0; AMP passed the Ultralytics safety check.
- Class-conditioned loss diagnostics are measured on a deterministic 32-image validation diagnostic batch, while AP uses the full 200-image clean validation split.
- Stage 1 must provide its own GT fallback because randomly initialized TAL assignments can be sparse or unstable.

