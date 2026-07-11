# Mini CS-DEM Clean-from-scratch Baseline

## Configuration

- Config: `configs/mini_csdem/clean_baseline.yaml`
- Split: `data_splits/mini_csdem_voc_seed0.json`
- Model: VOC20 YOLOv8n-style YAML, random initialization
- Epochs / image size / batch: 50 / 416 / 16
- Seed / workers / AMP: 0 / 0 / enabled
- Pretrained / resume / cache: false / false / false
- Clean validation is never perturbed.

Command:

```powershell
.\.venv\Scripts\python.exe runners\run_mini_csdem.py --mode baseline --config configs\mini_csdem\clean_baseline.yaml
```

## Results

| Metric | Value |
|---|---:|
| mAP50_target (person) | 0.308194 |
| mAP50_non_target | 0.082198 |
| mAP50_all | 0.093498 |
| worst_non_target_AP | 0.000434 |
| person-free non-target mAP50 | 0.082254 |
| person-cooccur non-target mAP50 | 0.136294 |

The 50-epoch run took 828.1 seconds. The selected best checkpoint SHA256 is `c5a731edd3ed75bca676e365bb5eb175ea404fb439bec3f9872b40999119ece7` and remains under ignored `outputs/mini_csdem/`.

## Per-class AP50

| Class | AP50 | Class | AP50 |
|---|---:|---|---:|
| aeroplane | 0.034366 | bicycle | 0.040027 |
| bird | 0.004675 | boat | 0.005049 |
| bottle | 0.006500 | bus | 0.030274 |
| car | 0.238151 | cat | 0.070064 |
| chair | 0.048741 | cow | 0.026804 |
| diningtable | 0.081348 | dog | 0.110903 |
| horse | 0.230293 | motorbike | 0.400276 |
| **person** | **0.308194** | pottedplant | 0.000434 |
| sheep | 0.000648 | sofa | 0.129870 |
| train | 0.086349 | tvmonitor | 0.016987 |

## Class-conditioned loss diagnostic

The following components use real Ultralytics TAL assignments on a deterministic 32-image clean validation diagnostic batch. Background negatives are excluded from the class-conditioned split.

| Component | Target | Non-target |
|---|---:|---:|
| classification loss | 2.172036 | 3.027947 |
| box loss | 2.332498 | 2.226160 |
| DFL loss | 2.619434 | 2.487270 |
| assigned positives | 474 | 567 |

## Artifacts and conclusion

- Metrics: `results/mini_csdem/clean_baseline_metrics.json`
- Per-class CSV: `results/mini_csdem/clean_baseline_per_class.csv`
- Training curve: `outputs/mini_csdem/train_runs/clean_baseline_seed0/results.csv` (ignored)
- Checkpoint: `outputs/mini_csdem/train_runs/clean_baseline_seed0/weights/best.pt` (ignored)

Stage 0 status: **PASS**. Mechanical checks passed and the clean baseline is reproducible. The low non-target AP floor means Stage 1–3 comparisons must report both mean non-target AP and worst-class changes, with explicit handling of classes already near zero.

