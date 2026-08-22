#!/usr/bin/env bash
set -Eeuo pipefail

: "${EXECUTION_COMMIT:?set EXECUTION_COMMIT to the reviewed full SHA}"

BRANCH="codex/tausb-sdh-dgcaip-cgr-e20-v2"
SOURCE_REPOSITORY="/root/autodl-tmp"
REQUIRED_STORAGE_ROOT="/root/autodl-tmp"
CHECKOUT="/root/autodl-tmp/tausb-dgcaip/checkouts/${EXECUTION_COMMIT}-mechanism"
SESSION="tausb-dgcaip-mechanism-s0-r2"
PYTHON_BIN="/root/miniconda3/bin/python"
# SHA256 of the LF-normalized config bytes in the reviewed Linux checkout.
EXPECTED_CONFIG_SHA256="a5de2322f40c090103895d869d5aeb528379ced58be285017d51a615a592119d"
ARTIFACT_ROOT="/root/autodl-tmp/tausb-dgcaip-runs/TAUSB-SDH-DGCAIP-S0-R2-MECHANISM"
CONTROL_ROOT="/root/autodl-tmp/tausb-dgcaip-control/TAUSB-SDH-DGCAIP-S0-R2-MECHANISM"
CACHE_ROOT="/root/autodl-tmp/tausb-dgcaip-cache/TAUSB-SDH-DGCAIP-S0-R2-MECHANISM"
TMP_ROOT="/root/autodl-tmp/tausb-dgcaip-tmp/TAUSB-SDH-DGCAIP-S0-R2-MECHANISM"
OUTER_LOG="/root/autodl-tmp/tausb-dgcaip-logs/TAUSB-SDH-DGCAIP-S0-R2-MECHANISM.log"
PRELAUNCH_FAILURE_LOG="/root/autodl-tmp/tausb-dgcaip-logs/TAUSB-SDH-DGCAIP-S0-R2-MECHANISM-prelaunch-failure.log"
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
      printf 'schema=tausb.dgcaip-mechanism-prelaunch-failure.v1\n'
      printf 'exit_code=%q\n' "${rc}"
      printf 'failed_line=%q\n' "${FAILED_LINE}"
      printf 'failed_command=%q\n' "${FAILED_COMMAND}"
      printf 'execution_commit=%q\n' "${EXECUTION_COMMIT}"
    } > "${PRELAUNCH_FAILURE_LOG}"
  fi
  echo "[DGCAIP-Mechanism-Launch] prelaunch failed with exit ${rc}; requesting shutdown" >&2
  sync || true
  /usr/bin/shutdown -h now || true
  exit "${rc}"
}

trap 'record_prelaunch_error "${LINENO}" "${BASH_COMMAND}"' ERR
trap shutdown_on_prelaunch_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mountpoint -q "${REQUIRED_STORAGE_ROOT}"
command -v tmux >/dev/null 2>&1
test -x /usr/bin/shutdown
git -C "${SOURCE_REPOSITORY}" rev-parse --is-inside-work-tree >/dev/null
test -x "${PYTHON_BIN}"
test -d "/root/autodl-tmp/ue_project/VOC_0712_Kaggle_Ready"
test -f "/root/autodl-tmp/ue_project/checkpoints/voc20_surrogate.pt"
test -f "/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-MECH/mechanism/p1_state.pt"
test -f "/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-MECH/mechanism/mechanism_metrics.json"
test -f "/root/tausb-sdh-runs/TAUSB-SDH-LFC-CICR-CGR-NLA-S0-r2/hiding/hiding_checkpoint.pt"
test -f "/root/autodl-tmp/tausb-dgcaip-runs/TAUSB-SDH-DGCAIP-S0-R1-D0/d0/d0_locator.json"
test ! -e "${CHECKOUT}"
test ! -e "${ARTIFACT_ROOT}"
test ! -e "${CONTROL_ROOT}"
test ! -e "${CACHE_ROOT}"
test ! -e "${TMP_ROOT}"
test ! -e "${OUTER_LOG}"
test ! -e "${PRELAUNCH_FAILURE_LOG}"
! tmux has-session -t "${SESSION}" 2>/dev/null

git -C "${SOURCE_REPOSITORY}" fetch origin "${BRANCH}"
git -C "${SOURCE_REPOSITORY}" cat-file -e "${EXECUTION_COMMIT}^{commit}"
git -C "${SOURCE_REPOSITORY}" merge-base --is-ancestor \
  "${EXECUTION_COMMIT}" "origin/${BRANCH}"
mkdir -p "$(dirname "${CHECKOUT}")" "$(dirname "${OUTER_LOG}")"
git -C "${SOURCE_REPOSITORY}" worktree add --detach "${CHECKOUT}" "${EXECUTION_COMMIT}"
test "$(git -C "${CHECKOUT}" rev-parse HEAD)" = "${EXECUTION_COMMIT}"
test -z "$(git -C "${CHECKOUT}" status --porcelain --untracked-files=all)"

WRAPPER="${CHECKOUT}/research_workspace/experiments/TAUSB-SDH-DGCAIP-S0-E20/pre_run/dgcaip_mechanism_controller_v1.sh"
CONFIG_PATH="${CHECKOUT}/ue_project/ue_framework/configs/tausb_sdh_dgcaip_mechanism_v2.yaml"
test -f "${WRAPPER}"
test -f "${CONFIG_PATH}"
test "$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')" = "${EXPECTED_CONFIG_SHA256}"

printf -v TMUX_COMMAND \
  'env PYTHON_BIN=%q REPOSITORY_ROOT=%q REQUIRED_STORAGE_ROOT=%q EXPECTED_COMMIT=%q EXPECTED_CONFIG_SHA256=%q CONFIG_PATH=%q ARTIFACT_ROOT=%q CONTROL_ROOT=%q CACHE_ROOT=%q TMP_ROOT=%q /bin/bash %q > %q 2>&1' \
  "${PYTHON_BIN}" "${CHECKOUT}" "${REQUIRED_STORAGE_ROOT}" \
  "${EXECUTION_COMMIT}" "${EXPECTED_CONFIG_SHA256}" "${CONFIG_PATH}" \
  "${ARTIFACT_ROOT}" "${CONTROL_ROOT}" "${CACHE_ROOT}" "${TMP_ROOT}" \
  "${WRAPPER}" "${OUTER_LOG}"

tmux new-session -d -s "${SESSION}" "${TMUX_COMMAND}"
trap - ERR EXIT INT TERM
echo "[DGCAIP-Mechanism-Launch] handed off to tmux session ${SESSION}"
echo "[DGCAIP-Mechanism-Launch] outer log: ${OUTER_LOG}"
echo "[DGCAIP-Mechanism-Launch] status: ${CONTROL_ROOT}/controller_status.json"
