#!/usr/bin/env bash
set -Eeuo pipefail

CODE_COMMIT="18304b96c45360cfba5168d97d21d2961a13f390"
PROJECT_ROOT="/root/tausb-malc-geometry-wt-v2/ue_project"
ARTIFACT_ROOT="/root/tausb-sirc-runs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1/geometry"
CONTROL_ROOT="/root/tausb-sirc-runs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1/control/geometry-seed0-18304b96-r1"
LOG_PATH="${CONTROL_ROOT}/geometry-seed0.log"
GUARD_STATE="${CONTROL_ROOT}/cost-guard-status.json"
READY_ROOT="${CONTROL_ROOT}/ready"
PYTHON_BIN="/root/miniconda3/bin/python"
CONFIG_PATH="ue_framework/configs/exp_voc_person_malc_grad_geometry_audit_v1.yaml"
TOOL_PATH="ue_framework/tools/probe_tausb_malc_geometry.py"
REMOTE_INPUT_AUDIT="/root/tausb-malc-geometry-input-audit.json"
PRIOR_INPUT_AUDIT="/root/tausb-sirc-runs/TAUSB-SIRC-MALC-CGR-MAP50-v2/mechanism/input_audit.json"
CURRENT_STAGE="preflight"
WATCHDOG_PID=""
RUNNER_PID="$$"

# Both the formal artifact root and this run-specific control root are fail-closed.
test ! -e "${ARTIFACT_ROOT}"
test ! -e "${CONTROL_ROOT}"
mkdir -p "${CONTROL_ROOT}"
touch "${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

write_guard_state() {
  local state="$1"
  local detail="$2"
  local tmp_path="${GUARD_STATE}.tmp"
  printf '{"state":"%s","stage":"%s","detail":"%s","code_commit":"%s","updated_at":"%s"}\n' \
    "${state}" "${CURRENT_STAGE}" "${detail}" "${CODE_COMMIT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${tmp_path}"
  mv "${tmp_path}" "${GUARD_STATE}"
}

snapshot_evidence() {
  local candidate
  mkdir -p "${READY_ROOT}"
  for candidate in \
    status.json \
    config_resolved.json \
    input_audit.json \
    prototype_geometry.json \
    gradient_geometry.json \
    microtrajectory.json \
    diagnostic_decision.json; do
    if [[ -f "${ARTIFACT_ROOT}/${candidate}" ]]; then
      cp "${ARTIFACT_ROOT}/${candidate}" "${READY_ROOT}/${candidate}"
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
    write_guard_state "completed" "geometry_probe_finished_shutdown_requested"
  else
    write_guard_state "failed" "geometry_probe_exit_${rc}_shutdown_requested"
  fi
  echo "[CostGuard] probe_exit=${rc}; requesting AutoDL shutdown"
  sync
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
    "${ARTIFACT_ROOT}/status.json" \
    "${ARTIFACT_ROOT}/config_resolved.json" \
    "${ARTIFACT_ROOT}/input_audit.json" \
    "${ARTIFACT_ROOT}/prototype_geometry.json" \
    "${ARTIFACT_ROOT}/gradient_geometry.json" \
    "${ARTIFACT_ROOT}/microtrajectory.json" \
    "${ARTIFACT_ROOT}/diagnostic_decision.json"; do
    if [[ -f "${candidate}" ]]; then
      value="$(stat -c %Y "${candidate}" 2>/dev/null || echo 0)"
      if [[ "${value}" =~ ^[0-9]+$ ]] && (( value > newest )); then
        newest="${value}"
      fi
    fi
  done
  echo "${newest}"
}

active_probe_cpu() {
  ps -eo pcpu=,args= | awk '
    /probe_tausb_malc_geometry.py/ && !/awk/ {sum += $1}
    END {printf "%d\n", sum + 0}
  '
}

cost_watchdog() {
  local last_progress
  local idle_since=0
  local now
  local observed
  local gpu_util
  local python_cpu
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
    python_cpu="$(active_probe_cpu)"
    [[ "${gpu_util}" =~ ^[0-9]+$ ]] || gpu_util=0
    [[ "${python_cpu}" =~ ^[0-9]+$ ]] || python_cpu=0
    if (( gpu_util <= 5 && python_cpu <= 5 )); then
      now="$(date +%s)"
      if (( idle_since == 0 )); then
        idle_since="${now}"
      elif (( now - idle_since >= 600 )); then
        echo "[CostGuard] no log/artifact progress and idle GPU/CPU for 10 minutes"
        write_guard_state "hang_guard_triggered" "10m_no_progress_gpu_cpu_idle"
        kill -TERM "${RUNNER_PID}" 2>/dev/null || true
        sleep 10
        snapshot_evidence || true
        sync
        /usr/bin/shutdown || true
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
test "$(git rev-parse HEAD)" = "${CODE_COMMIT}"
test "$(git ls-files ue_framework configs | wc -l | tr -d ' ')" = "131"
git diff-index --quiet HEAD -- ue_framework configs
test -z "$(git diff --name-only -- ue_framework configs)"
test -z "$(git diff --cached --name-only -- ue_framework configs)"
unexpected_untracked="$(
  git ls-files --others --exclude-standard -- ue_framework configs \
    | grep -Ev '(^|/)__pycache__/.*\.pyc$' || true
)"
test -z "${unexpected_untracked}"
test ! -e "${ARTIFACT_ROOT}"
test -x "${PYTHON_BIN}"
test -f "${CONFIG_PATH}"
test -f "${TOOL_PATH}"
test -f "${REMOTE_INPUT_AUDIT}"
test -f "${PRIOR_INPUT_AUDIT}"
test -x /usr/bin/shutdown

"${PYTHON_BIN}" - "${REMOTE_INPUT_AUDIT}" "${PRIOR_INPUT_AUDIT}" <<'PY'
import json
import sys

current = json.load(open(sys.argv[1], encoding="utf-8"))
prior = json.load(open(sys.argv[2], encoding="utf-8"))
expected = {
    "label_sha256": "0c8b6f6424061bc31b84ddf42b7370dcbd074f26805433d0ba275c24815e3248",
    "shared_split_sha256": "e2542517af00830147117582d69ff15a62fbeae1f8583bf0c9d01fbff120cae1",
    "source_manifest_sha256": "3a13b0f38b06006fd7f68ae03c7206b4b047d4b6129ee7357b05b966641d47af",
    "surrogate_checkpoint_sha256": "8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89",
}
assert current["pass"] is True
assert current["artifact_root_fresh"] is True
assert current["calibration_images"] == 64
assert current["heldout_images"] == 96
for key, value in expected.items():
    assert current["actual"][key] == value, (key, current["actual"][key])
assert prior["semantic_bank_hash"] == "0b8a94efc55155bea20a1ec799bfac14c8a6f11fd6530538f3e0437b37c0dd4b"
assert prior["c2lm_basis_hash"] == "8350c0a608150839c98a8dad8db862d0c9dfaeca4714f05d1714afac0f30cfa5"
print("[Harness] frozen input audit PASS")
PY

"${PYTHON_BIN}" - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
name = torch.cuda.get_device_name(0)
assert "4090" in name, name
print("[Harness] CUDA device PASS:", name)
PY

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

cost_watchdog &
WATCHDOG_PID="$!"

mark_stage "geometry_probe" "surrogate_only_no_eot_no_victim_no_materialization"
"${PYTHON_BIN}" -u "${TOOL_PATH}" \
  --config "${CONFIG_PATH}" \
  --device 0

mark_stage "evidence_snapshot" "copy_minimal_geometry_results_before_shutdown"
snapshot_evidence
mark_stage "completed" "geometry_probe_and_evidence_snapshot_complete"
