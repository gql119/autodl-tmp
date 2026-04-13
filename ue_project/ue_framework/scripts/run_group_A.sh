#!/usr/bin/env bash
set -euo pipefail

CONFIG="ue_framework/configs/exp_voc_person_formal.yaml"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="./runs_formal/logs"

mkdir -p "${LOG_DIR}"

for method in ours_mask rem_mask; do
  for seed in 0 1 2; do
    log_file="${LOG_DIR}/${method}_steps40_seed${seed}.log"
    echo "[run_group_A] method=${method} steps=40 seed=${seed} gpu=${GPU_ID} log=${log_file}"
    python -u ue_framework/launch_one.py \
      --config "${CONFIG}" \
      --method "${method}" \
      --steps 40 \
      --seed "${seed}" \
      --stage all \
      --gpu_id "${GPU_ID}" \
      > "${log_file}" 2>&1
  done
done

echo "[run_group_A] done"
