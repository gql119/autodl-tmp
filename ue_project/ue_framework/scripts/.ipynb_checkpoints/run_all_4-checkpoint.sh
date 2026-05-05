#!/usr/bin/env bash
set -euo pipefail

# [配置核对] 目前读取的是 tausb_formal。如果你想跑 ours_mask 实验，请将文件名改为 exp_voc_person_formal.yaml
CONFIG="ue_framework/configs/exp_voc_person_tausb_formal.yaml"
GPU_ID="${GPU_ID:-0}"

# [致命修复] 强制将全局输出目录指向 50GB 数据盘！
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/ue_project/runs}"
LOG_DIR="${RUN_ROOT}/logs"

mkdir -p "${LOG_DIR}"

methods=(tausb_mask)
seeds=(0)

for method in "${methods[@]}"; do
  for seed in "${seeds[@]}"; do
    log_file="${LOG_DIR}/${method}_steps40_seed${seed}_xr1.log"
    echo "[run_all_4] method=${method} steps=40 seed=${seed} gpu=${GPU_ID}"
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

echo "[run_all_4] done"
