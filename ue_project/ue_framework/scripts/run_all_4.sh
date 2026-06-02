#!/usr/bin/env bash
set -euo pipefail

# [配置核对] 目前读取的是 tausb_formal。如果你想跑 ours_mask 实验，请将文件名改为 exp_voc_person_formal.yaml
#CONFIG="ue_framework/configs/exp_voc_person_tausb_formal.yaml"
CONFIG="ue_framework/configs/exp_voc_person_tausb_fhml2_late_repair_full.yaml"
GPU_ID="${GPU_ID:-0}"

# [致命修复] 强制将全局输出目录指向 50GB 数据盘！
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/ue_project/runs}"
LOG_DIR="${RUN_ROOT}/logs"

mkdir -p "${LOG_DIR}"

methods=(tausb_mask)
seeds=(0)

#stages=(train_victim evaluate aggregate)
stages=(generate_poisoned_dataset train_victim evaluate aggregate)
#stages=(generate_poisoned_dataset)
for method in "${methods[@]}"; do
  for seed in "${seeds[@]}"; do
    for stage in "${stages[@]}"; do
      log_file="${LOG_DIR}/${method}_steps40_seed${seed}_${stage}.log"

      echo "[run_remaining] method=${method} steps=40 seed=${seed} stage=${stage} gpu=${GPU_ID}"

      python -u ue_framework/launch_one.py \
        --config "${CONFIG}" \
        --method "${method}" \
        --steps 40 \
        --seed "${seed}" \
        --stage "${stage}" \
        --gpu_id "${GPU_ID}" \
        > "${log_file}" 2>&1
    done
  done
done

echo "[run_all_4] done"
