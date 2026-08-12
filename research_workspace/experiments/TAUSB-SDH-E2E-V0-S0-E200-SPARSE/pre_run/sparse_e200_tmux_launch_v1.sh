#!/usr/bin/env bash
set -Eeuo pipefail

BRANCH="codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3"
EXECUTION_COMMIT="36f74cab2222f41cb1f206b42db3118237f18a52"
SOURCE_REPOSITORY="/root/autodl-tmp"
REQUIRED_STORAGE_ROOT="/root/autodl-tmp"
CHECKOUT="/root/autodl-tmp/tausb-sdh/checkouts/${EXECUTION_COMMIT}"
SESSION="tausb-sdh-e200-s0-r1"
PYTHON_BIN="/root/miniconda3/bin/python"
SKIP_GIT_FETCH="${SKIP_GIT_FETCH:-0}"
MECHANISM_ROOT="/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-MECH"
DATASET_ROOT="/root/autodl-tmp/ue_project/VOC_0712_Kaggle_Ready"
BINDING_ROOT="/root/autodl-tmp/tausb-sdh/binding/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1"
RUN_ROOT_PREFIX="/root/autodl-tmp/tausb-sdh/runs/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1"
CONTROL_ROOT="/root/autodl-tmp/tausb-sdh/control/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1"
LOG_ROOT="/root/autodl-tmp/tausb-sdh/logs/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1"
COMPARISON_ROOT="/root/autodl-tmp/tausb-sdh/comparison/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1"
CACHE_ROOT="/root/autodl-tmp/tausb-sdh/cache/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1"
TMP_ROOT="/root/autodl-tmp/tausb-sdh/tmp/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1"
OUTER_LOG="/root/autodl-tmp/tausb-sdh/launch/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1.log"
PRELAUNCH_FAILURE_LOG="/root/autodl-tmp/tausb-sdh/launch/TAUSB-SDH-E2E-V0-S0-E200-SPARSE-R1-prelaunch-failure.log"
FAILED_LINE="unknown"
FAILED_COMMAND="unknown"

record_prelaunch_error() {
  local rc=$?
  FAILED_LINE="$1"
  FAILED_COMMAND="$2"
  return "${rc}"
}

shutdown_on_prelaunch_failure() {
  local rc=$?
  trap - ERR EXIT INT TERM
  if mountpoint -q "${REQUIRED_STORAGE_ROOT}"; then
    mkdir -p "$(dirname "${PRELAUNCH_FAILURE_LOG}")"
    {
      printf 'schema=tausb.sdh-e200-prelaunch-failure.v1\n'
      printf 'exit_code=%q\n' "${rc}"
      printf 'failed_line=%q\n' "${FAILED_LINE}"
      printf 'failed_command=%q\n' "${FAILED_COMMAND}"
      printf 'execution_commit=%q\n' "${EXECUTION_COMMIT}"
    } > "${PRELAUNCH_FAILURE_LOG}"
  fi
  echo "[SparseE200Launch] prelaunch failed with exit ${rc}; requesting shutdown" >&2
  sync || true
  /usr/bin/shutdown -h now || true
  exit "${rc}"
}

trap 'record_prelaunch_error "${LINENO}" "${BASH_COMMAND}"' ERR
trap shutdown_on_prelaunch_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mountpoint -q "${REQUIRED_STORAGE_ROOT}"
git -C "${SOURCE_REPOSITORY}" rev-parse --is-inside-work-tree >/dev/null
test -x "${PYTHON_BIN}"
test -d "${DATASET_ROOT}"
test -f "${MECHANISM_ROOT}/mechanism/p1_feasibility_sdh_state.pt"
test -f "${MECHANISM_ROOT}/mechanism/p1_state.pt"
test ! -e "${CHECKOUT}"
test ! -e "${BINDING_ROOT}"
test ! -e "${RUN_ROOT_PREFIX}-E200-C0"
test ! -e "${RUN_ROOT_PREFIX}-E200-M1"
test ! -e "${CONTROL_ROOT}"
test ! -e "${LOG_ROOT}"
test ! -e "${COMPARISON_ROOT}"
test ! -e "${CACHE_ROOT}"
test ! -e "${TMP_ROOT}"
test ! -e "${OUTER_LOG}"
test ! -e "${PRELAUNCH_FAILURE_LOG}"
! tmux has-session -t "${SESSION}" 2>/dev/null

git -C "${SOURCE_REPOSITORY}" cat-file -e "${EXECUTION_COMMIT}^{commit}"
if [[ "${SKIP_GIT_FETCH}" == "1" ]]; then
  echo "[SparseE200Launch] offline launch: using hash-verified cached commit ${EXECUTION_COMMIT}"
elif [[ "${SKIP_GIT_FETCH}" == "0" ]]; then
  git -C "${SOURCE_REPOSITORY}" fetch origin "${BRANCH}"
  git -C "${SOURCE_REPOSITORY}" merge-base --is-ancestor \
    "${EXECUTION_COMMIT}" "origin/${BRANCH}"
else
  echo "[SparseE200Launch] SKIP_GIT_FETCH must be 0 or 1" >&2
  exit 20
fi
mkdir -p "$(dirname "${CHECKOUT}")" "$(dirname "${OUTER_LOG}")"
git -C "${SOURCE_REPOSITORY}" worktree add --detach "${CHECKOUT}" "${EXECUTION_COMMIT}"
test "$(git -C "${CHECKOUT}" rev-parse HEAD)" = "${EXECUTION_COMMIT}"
test -z "$(git -C "${CHECKOUT}" status --porcelain --untracked-files=all)"

WRAPPER="${CHECKOUT}/research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E200-SPARSE/pre_run/sparse_e200_controller_data_disk_v1.sh"
MECHANISM_CONFIG="${CHECKOUT}/ue_project/ue_framework/configs/tausb_sdh_e2e_v0_mechanism.yaml"
BASE_CONFIG="${CHECKOUT}/ue_project/ue_framework/configs/exp_voc_person_sdh_lfc_cicr_cgr_nla_map50_v3.yaml"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = \
  "06a7f652fbb12d4a0c9bd896ca836a063b05a4b9d826cd64abd25548287c3684"

printf -v TMUX_COMMAND \
  'env PYTHON_BIN=%q DEVICE=%q REPOSITORY_ROOT=%q REQUIRED_STORAGE_ROOT=%q EXPECTED_COMMIT=%q MECHANISM_ROOT=%q MECHANISM_CONFIG=%q BASE_CONFIG=%q DATASET_ROOT=%q BINDING_ROOT=%q RUN_ROOT_PREFIX=%q CONTROL_ROOT=%q LOG_ROOT=%q COMPARISON_ROOT=%q CACHE_ROOT=%q TMP_ROOT=%q /bin/bash %q > %q 2>&1' \
  "${PYTHON_BIN}" "0" "${CHECKOUT}" "${REQUIRED_STORAGE_ROOT}" \
  "${EXECUTION_COMMIT}" "${MECHANISM_ROOT}" "${MECHANISM_CONFIG}" \
  "${BASE_CONFIG}" "${DATASET_ROOT}" "${BINDING_ROOT}" \
  "${RUN_ROOT_PREFIX}" "${CONTROL_ROOT}" "${LOG_ROOT}" \
  "${COMPARISON_ROOT}" "${CACHE_ROOT}" "${TMP_ROOT}" "${WRAPPER}" \
  "${OUTER_LOG}"

tmux new-session -d -s "${SESSION}" "${TMUX_COMMAND}"
trap - ERR EXIT INT TERM
echo "[SparseE200Launch] handed off to tmux session ${SESSION}"
echo "[SparseE200Launch] outer log: ${OUTER_LOG}"
echo "[SparseE200Launch] inspect: tmux capture-pane -pt ${SESSION} -S -120"
echo "[SparseE200Launch] status: ${CONTROL_ROOT}/controller_status.json"
