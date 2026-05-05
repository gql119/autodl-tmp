#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
RUN_ROOT="${RUN_ROOT:-/root/ue_project/runs}"
LOG_DIR="${RUN_ROOT}/logs"
STAGE="${STAGE:-all}"
STEPS="${STEPS:-40}"

# 默认全方法顺序执行；如需子集可通过 METHODS 覆盖，例如：
# METHODS="tausb_mask tap_mask lsp_mask" bash ue_framework/scripts/run_all_4.sh
METHODS_STR="${METHODS:-tap_mask lsp_mask em_mask rem_mask}"
SEEDS_STR="${SEEDS:-0}"

# 各方法默认配置（可被环境变量覆盖）
#CONFIG_TAUSB="${CONFIG_TAUSB:-ue_framework/configs/exp_voc_person_tausb_formal.yaml}"
CONFIG_TAP="${CONFIG_TAP:-ue_framework/configs/exp_voc_person_tap_mask.yaml}"
CONFIG_LSP="${CONFIG_LSP:-ue_framework/configs/exp_voc_person_lsp_mask.yaml}"
CONFIG_EM="${CONFIG_EM:-ue_framework/configs/exp_voc_person_em_mask.yaml}"
CONFIG_REM="${CONFIG_REM:-ue_framework/configs/exp_voc_person_rem_mask.yaml}"
#CONFIG_OURS="${CONFIG_OURS:-ue_framework/configs/exp_voc_person_formal.yaml}"

mkdir -p "${LOG_DIR}"

get_config_for_method() {
  local method="$1"
  case "${method}" in
    #tausb_mask) echo "${CONFIG_TAUSB}" ;;
    tap_mask)   echo "${CONFIG_TAP}" ;;
    lsp_mask)   echo "${CONFIG_LSP}" ;;
    #em_bbox)    echo "${CONFIG_EM}" ;;
    em_mask)    echo "${CONFIG_EM}" ;;
    rem_mask)   echo "${CONFIG_REM}" ;;
    #ours_mask)  echo "${CONFIG_OURS}" ;;
    *)
      echo "[run_all_4] Unsupported method: ${method}" >&2
      return 1
      ;;
  esac
}

for method in ${METHODS_STR}; do
  config_file="$(get_config_for_method "${method}")"
  if [[ ! -f "${config_file}" ]]; then
    echo "[run_all_4] Config not found for ${method}: ${config_file}" >&2
    exit 1
  fi

  for seed in ${SEEDS_STR}; do
    log_file="${LOG_DIR}/${method}_steps${STEPS}_seed${seed}.log"
    echo "[run_all_4] method=${method} steps=${STEPS} seed=${seed} gpu=${GPU_ID} stage=${STAGE} config=${config_file}"
    python -u ue_framework/launch_one.py \
      --config "${config_file}" \
      --method "${method}" \
      --steps "${STEPS}" \
      --seed "${seed}" \
      --stage "${STAGE}" \
      --gpu_id "${GPU_ID}" \
      > "${log_file}" 2>&1
  done
done

echo "[run_all_4] done"
