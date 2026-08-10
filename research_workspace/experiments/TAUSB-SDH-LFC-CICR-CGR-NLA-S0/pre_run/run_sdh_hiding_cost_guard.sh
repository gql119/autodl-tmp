#!/usr/bin/env bash
set -Eeuo pipefail

CODE_COMMIT="e3f674497569087a79dd3782fcdfbabd4e7c8d04"
REPOSITORY_ROOT="/root/tausb-sdh-checkouts/6f4d5d8-worktree"
PROJECT_ROOT="${REPOSITORY_ROOT}/ue_project"
ARTIFACT_ROOT="/root/tausb-sdh-runs/TAUSB-SDH-LFC-CICR-CGR-NLA-S0"
CONTROL_ROOT="/root/tausb-sdh-control/TAUSB-SDH-LFC-CICR-CGR-NLA-S0/hiding-e3f6744-r1"
LOG_PATH="${CONTROL_ROOT}/hiding.log"
GUARD_STATE="${CONTROL_ROOT}/cost-guard-status.json"
READY_ROOT="${CONTROL_ROOT}/ready"
PYTHON_BIN="/root/miniconda3/bin/python"
CONFIG_PATH="ue_framework/configs/tausb_sdh_mechanism_v3.yaml"
TOOL_PATH="ue_framework/tools/run_tausb_sdh.py"
TOOL_MODULE="ue_framework.tools.run_tausb_sdh"
CURRENT_STAGE="preflight"
WATCHDOG_PID=""
RUNNER_PID="$$"

preflight_shutdown() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    echo "[CostGuard] preflight_exit=${rc}; requesting AutoDL shutdown" >&2
    sync || true
    /usr/bin/shutdown || true
  fi
  exit "${rc}"
}
trap preflight_shutdown EXIT

test ! -e "${ARTIFACT_ROOT}"
test ! -e "${CONTROL_ROOT}"
mkdir -p "${CONTROL_ROOT}"
touch "${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

write_guard_state() {
  local state="$1"
  local detail="$2"
  local temporary="${GUARD_STATE}.tmp"
  printf '{"state":"%s","stage":"%s","detail":"%s","code_commit":"%s","updated_at":"%s"}\n' \
    "${state}" "${CURRENT_STAGE}" "${detail}" "${CODE_COMMIT}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${temporary}"
  mv "${temporary}" "${GUARD_STATE}"
}

snapshot_evidence() {
  local candidate
  mkdir -p "${READY_ROOT}"
  for candidate in \
    "${ARTIFACT_ROOT}/status_hiding.json" \
    "${ARTIFACT_ROOT}/hiding/hiding_metrics.json" \
    "${ARTIFACT_ROOT}/hiding/split_manifest.json"; do
    if [[ -f "${candidate}" ]]; then
      cp "${candidate}" "${READY_ROOT}/$(basename "${candidate}")"
    fi
  done
}

finish() {
  local rc=$?
  trap - EXIT
  if [[ -n "${WATCHDOG_PID}" ]]; then
    kill "${WATCHDOG_PID}" 2>/dev/null || true
  fi
  snapshot_evidence || true
  if [[ ${rc} -eq 0 ]]; then
    write_guard_state "completed" "hiding_pilot_finished_shutdown_requested" || true
  else
    write_guard_state "failed" "hiding_pilot_exit_${rc}_shutdown_requested" || true
  fi
  echo "[CostGuard] hiding_exit=${rc}; requesting AutoDL shutdown"
  sync || true
  /usr/bin/shutdown || true
  exit "${rc}"
}
trap finish EXIT
trap 'exit 130' INT TERM

progress_mtime() {
  local newest=0
  local candidate
  local value
  for candidate in \
    "${LOG_PATH}" \
    "${GUARD_STATE}" \
    "${ARTIFACT_ROOT}/status_hiding.json" \
    "${ARTIFACT_ROOT}/hiding/hiding_metrics.json" \
    "${ARTIFACT_ROOT}/hiding/split_manifest.json"; do
    if [[ -f "${candidate}" ]]; then
      value="$(stat -c %Y "${candidate}" 2>/dev/null || echo 0)"
      if [[ "${value}" =~ ^[0-9]+$ ]] && (( value > newest )); then
        newest="${value}"
      fi
    fi
  done
  echo "${newest}"
}

active_pilot_cpu() {
  ps -eo pcpu=,args= | awk '
    /[r]un_tausb_sdh.py/ && /--stage hiding/ {sum += $1}
    END {printf "%d\n", sum + 0}
  '
}

cost_watchdog() {
  local last_progress
  local idle_since=0
  local observed
  local gpu_util
  local python_cpu
  local now
  last_progress="$(progress_mtime)"
  while kill -0 "${RUNNER_PID}" 2>/dev/null; do
    sleep 60
    observed="$(progress_mtime)"
    if [[ "${observed}" != "${last_progress}" ]]; then
      last_progress="${observed}"
      idle_since=0
      continue
    fi
    gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || echo 0)"
    python_cpu="$(active_pilot_cpu)"
    [[ "${gpu_util}" =~ ^[0-9]+$ ]] || gpu_util=0
    [[ "${python_cpu}" =~ ^[0-9]+$ ]] || python_cpu=0
    if (( gpu_util <= 5 && python_cpu <= 5 )); then
      now="$(date +%s)"
      if (( idle_since == 0 )); then
        idle_since="${now}"
      elif (( now - idle_since >= 600 )); then
        echo "[CostGuard] 10 minutes without progress and with idle GPU/CPU"
        write_guard_state "hang_guard_triggered" "10m_no_progress_gpu_cpu_idle"
        kill -TERM "${RUNNER_PID}" 2>/dev/null || true
        return
      fi
    else
      idle_since=0
    fi
  done
}

mark_stage() {
  CURRENT_STAGE="$1"
  write_guard_state "running" "$2"
  echo "[Harness] stage=${CURRENT_STAGE} detail=$2 time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

cd "${PROJECT_ROOT}"
test "$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD)" = "${CODE_COMMIT}"
git -C "${REPOSITORY_ROOT}" diff-index --quiet HEAD -- ue_project/ue_framework research_workspace/sources
test -z "$(git -C "${REPOSITORY_ROOT}" diff --name-only -- ue_project/ue_framework research_workspace/sources)"
test -z "$(git -C "${REPOSITORY_ROOT}" diff --cached --name-only -- ue_project/ue_framework research_workspace/sources)"
test ! -e "${ARTIFACT_ROOT}"
test -x "${PYTHON_BIN}"
test -x /usr/bin/timeout
test -x /usr/bin/shutdown
test -f "${CONFIG_PATH}"
test -f "${TOOL_PATH}"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

cost_watchdog &
WATCHDOG_PID="$!"

mark_stage "hiding_pilot" "single_secret_no_victim_no_mechanism_no_eot_no_jnd"
/usr/bin/timeout --signal=TERM --kill-after=30s 1200s \
  "${PYTHON_BIN}" -u -m "${TOOL_MODULE}" \
  --config "${CONFIG_PATH}" \
  --stage hiding

mark_stage "evidence_snapshot" "copy_minimal_hiding_results_before_shutdown"
snapshot_evidence
mark_stage "completed" "hiding_pilot_and_evidence_snapshot_complete"
