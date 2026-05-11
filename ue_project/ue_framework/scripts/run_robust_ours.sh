#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/ue_project}"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
STEPS="${STEPS:-40}"
STAGE="${STAGE:-all}"

RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/ue_project/runs_robust}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

METHOD="tausb_mask"
SEEDS=(${SEEDS:-0})

CONFIGS=(
  "ue_framework/configs/exp_ours_robust_blur3.yaml"
  "ue_framework/configs/exp_ours_robust_jpeg50.yaml"
  "ue_framework/configs/exp_ours_robust_jpeg30.yaml"
)

for config_file in "${CONFIGS[@]}"; do
  if [[ ! -f "${config_file}" ]]; then
    echo "[run_robust] Config not found: ${config_file}" >&2
    echo "[run_robust] Current dir: $(pwd)" >&2
    echo "[run_robust] Existing robust configs:" >&2
    find ue_framework/configs -name "exp_ours_robust*.yaml" >&2 || true
    exit 1
  fi

  config_name="$(basename "${config_file}" .yaml)"

  for seed in "${SEEDS[@]}"; do
    log_file="${LOG_DIR}/${config_name}_${METHOD}_steps${STEPS}_seed${seed}_${STAGE}.log"

    echo "[run_robust] config=${config_file}"
    echo "[run_robust] method=${METHOD} steps=${STEPS} seed=${seed} gpu=${GPU_ID} stage=${STAGE}"
    echo "[run_robust] log=${log_file}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u ue_framework/launch_one.py \
      --config "${config_file}" \
      --method "${METHOD}" \
      --steps "${STEPS}" \
      --seed "${seed}" \
      --stage "${STAGE}" \
      > "${log_file}" 2>&1
  done
done

echo "[run_robust] done"