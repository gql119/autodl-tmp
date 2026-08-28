#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SHUTDOWN_BIN="${SHUTDOWN_BIN:-/usr/bin/shutdown}"
HARD_CAP_SECONDS=480
MAX_ARTIFACT_BYTES=104857600
STARTED_SECONDS="$(date +%s)"
CONTROL_ROOT_VALIDATED=0
CURRENT_STAGE="preflight"

shutdown_once() {
  local rc=$?
  trap - EXIT INT TERM
  if mountpoint -q "${REQUIRED_STORAGE_ROOT:-/missing}"; then
    local terminal_root="${REQUIRED_STORAGE_ROOT}/tausb-dgcaip-control/fallback"
    if [[ "${CONTROL_ROOT_VALIDATED}" -eq 1 ]]; then
      terminal_root="${CONTROL_ROOT}"
    fi
    mkdir -p "${terminal_root}"
    printf '{"schema":"tausb.p1-det-resize-wrapper-terminal.v1","exit_code":%d,"stage":"%s","shutdown_requested":true}\n' \
      "${rc}" "${CURRENT_STAGE}" > "${terminal_root}/wrapper_terminal.json"
  fi
  echo "[P1-DET-RESIZE] controller_exit=${rc}; requesting AutoDL shutdown"
  sync || true
  "${SHUTDOWN_BIN}" -h now || true
  exit "${rc}"
}

trap shutdown_once EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

: "${REPOSITORY_ROOT:?reviewed detached checkout is required}"
: "${REQUIRED_STORAGE_ROOT:?AutoDL data-disk root is required}"
: "${EXPECTED_COMMIT:?reviewed commit is required}"
: "${EXPECTED_CONFIG_SHA256:?reviewed config hash is required}"
: "${CONFIG_PATH:?reviewed resize-fix config is required}"
: "${ARTIFACT_ROOT:?unique artifact root is required}"
: "${CONTROL_ROOT:?unique control root is required}"
: "${CACHE_ROOT:?cache root is required}"
: "${TMP_ROOT:?temporary root is required}"

command -v timeout >/dev/null 2>&1
command -v nvidia-smi >/dev/null 2>&1
test -x "${PYTHON_BIN}"
mountpoint -q "${REQUIRED_STORAGE_ROOT}"

REPOSITORY_ROOT="$(realpath -m "${REPOSITORY_ROOT}")"
REQUIRED_STORAGE_ROOT="$(realpath -m "${REQUIRED_STORAGE_ROOT}")"
CONFIG_PATH="$(realpath -m "${CONFIG_PATH}")"
ARTIFACT_ROOT="$(realpath -m "${ARTIFACT_ROOT}")"
CONTROL_ROOT="$(realpath -m "${CONTROL_ROOT}")"
CACHE_ROOT="$(realpath -m "${CACHE_ROOT}")"
TMP_ROOT="$(realpath -m "${TMP_ROOT}")"

for growing_path in \
  "${REPOSITORY_ROOT}" "${ARTIFACT_ROOT}" "${CONTROL_ROOT}" \
  "${CACHE_ROOT}" "${TMP_ROOT}"; do
  case "${growing_path}" in
    "${REQUIRED_STORAGE_ROOT}"|"${REQUIRED_STORAGE_ROOT}"/*) ;;
    *) echo "[P1-DET-RESIZE] path outside data disk: ${growing_path}" >&2; exit 20 ;;
  esac
done
CONTROL_ROOT_VALIDATED=1

test "$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${REPOSITORY_ROOT}" status --porcelain --untracked-files=all)"
test -f "${CONFIG_PATH}"
test "$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')" = "${EXPECTED_CONFIG_SHA256}"
test ! -e "${ARTIFACT_ROOT}"
test ! -e "${CONTROL_ROOT}"

mkdir -p "${CONTROL_ROOT}" "${CACHE_ROOT}/xdg" "${CACHE_ROOT}/torch" \
  "${CACHE_ROOT}/yolo" "${CACHE_ROOT}/cuda" "${CACHE_ROOT}/matplotlib" \
  "${CACHE_ROOT}/huggingface" "${TMP_ROOT}"
export TMPDIR="${TMP_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export YOLO_CONFIG_DIR="${CACHE_ROOT}/yolo"
export CUDA_CACHE_PATH="${CACHE_ROOT}/cuda"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export HF_HOME="${CACHE_ROOT}/huggingface"

printf '{"schema":"tausb.p1-det-resize-controller.v1","status":"running","execution_commit":"%s","config_sha256":"%s","hard_cap_seconds":480}\n' \
  "${EXPECTED_COMMIT}" "${EXPECTED_CONFIG_SHA256}" \
  > "${CONTROL_ROOT}/controller_status.json"

cd "${REPOSITORY_ROOT}/ue_project"
"${PYTHON_BIN}" - "${CONFIG_PATH}" "${ARTIFACT_ROOT}" <<'PY'
import pathlib
import sys

import yaml

from ue_framework.methods.sdh_experiment import validate_sdh_experiment_config


config_path = pathlib.Path(sys.argv[1])
artifact_root = pathlib.Path(sys.argv[2])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
validate_sdh_experiment_config(config)
assert config["spec"]["spec_id"] == "TAUSB-SDH-DGCAIP-P1-DET-RESIZE-FIX-v1"
assert config["audit"]["normal_lanes"] == ["reset"]
assert config["audit"]["strict_lanes"] == ["fresh"]
assert config["audit"]["zero_parameter_updates"] is True
assert config["audit"]["total_hard_cap_seconds"] == 480
assert pathlib.Path(config["runtime"]["artifact_root"]) == artifact_root
PY

remaining_seconds() {
  echo "$(( HARD_CAP_SECONDS - ($(date +%s) - STARTED_SECONDS) ))"
}

run_budgeted() {
  local seconds="$1"
  shift
  local remaining
  remaining="$(remaining_seconds)"
  if [[ "${remaining}" -le 0 ]]; then
    return 124
  fi
  if [[ "${seconds}" -gt "${remaining}" ]]; then
    seconds="${remaining}"
  fi
  timeout --signal=TERM --kill-after=15s "${seconds}s" "$@"
}

run_rc=0
CURRENT_STAGE="G0_resize_microprobe"
set +e
run_budgeted 60 env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "${PYTHON_BIN}" -u -m ue_framework.tools.run_deterministic_resize_probe \
    --config "${CONFIG_PATH}" \
    --output "${CONTROL_ROOT}/resize_operator_probe.json" \
    --iterations 32
run_rc=$?
set -e

if [[ "${run_rc}" -eq 0 ]]; then
  CURRENT_STAGE="G1_normal_reset"
  set +e
  run_budgeted "$(remaining_seconds)" env -u CUBLAS_WORKSPACE_CONFIG \
    "${PYTHON_BIN}" -u -m ue_framework.tools.run_p1_determinism_audit \
      --config "${CONFIG_PATH}" --mode normal
  run_rc=$?
  set -e
fi
if [[ "${run_rc}" -eq 0 ]]; then
  CURRENT_STAGE="G1_strict_fresh"
  set +e
  run_budgeted "$(remaining_seconds)" env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "${PYTHON_BIN}" -u -m ue_framework.tools.run_p1_determinism_audit \
      --config "${CONFIG_PATH}" --mode strict
  run_rc=$?
  set -e
fi
if [[ "${run_rc}" -eq 0 ]]; then
  CURRENT_STAGE="G1_summarize"
  set +e
  run_budgeted "$(remaining_seconds)" \
    "${PYTHON_BIN}" -u -m ue_framework.tools.run_p1_determinism_audit \
      --config "${CONFIG_PATH}" --mode summarize
  run_rc=$?
  set -e
fi
if [[ "${run_rc}" -eq 0 ]]; then
  CURRENT_STAGE="G1_gate"
  set +e
  "${PYTHON_BIN}" - "${ARTIFACT_ROOT}/determinism_audit_summary.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
decision = payload["decision"]
assert decision["mechanical_pass"] is True
assert decision["label"] == "strict_replay_pass"
assert decision["strict_fresh_bitwise_pass"] is True
PY
  run_rc=$?
  set -e
fi
if [[ "${run_rc}" -eq 0 ]]; then
  CURRENT_STAGE="G2_writeback_smoke"
  set +e
  run_budgeted "$(remaining_seconds)" env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "${PYTHON_BIN}" -u -m ue_framework.tools.run_p1_resize_repair_writeback \
      --config "${CONFIG_PATH}"
  run_rc=$?
  set -e
fi

CURRENT_STAGE="summarize"
set +e
gate_label="$(${PYTHON_BIN} - \
  "${CONTROL_ROOT}/resize_operator_probe.json" \
  "${ARTIFACT_ROOT}/determinism_audit_summary.json" \
  "${ARTIFACT_ROOT}/writeback_smoke.json" \
  "${CONTROL_ROOT}/repair_gate_summary.json" \
  "${run_rc}" <<'PY'
import json
import pathlib
import sys

probe_path, audit_path, writeback_path, output_path = map(pathlib.Path, sys.argv[1:5])
run_rc = int(sys.argv[5])
label = "infra_failure"
checks = {}
probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.is_file() else {}
audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
writeback = (
    json.loads(writeback_path.read_text(encoding="utf-8"))
    if writeback_path.is_file()
    else {}
)
checks["resize_probe_pass"] = probe.get("status") == "passed"
checks["strict_replay_pass"] = (
    audit.get("decision", {}).get("label") == "strict_replay_pass"
)
checks["writeback_pass"] = writeback.get("passed") is True
if run_rc == 124:
    label = "performance_gate_failed"
elif probe and probe.get("status") != "passed":
    error = str(probe.get("error", "")).lower()
    label = (
        "new_cuda_nondeterministic_operator"
        if "deterministic" in error
        else "resize_forward_or_gradient_mismatch"
    )
elif probe and float(probe.get("benchmark_seconds", 0.0)) > 60.0:
    label = "performance_gate_failed"
elif audit and audit.get("decision", {}).get("label") != "strict_replay_pass":
    label = str(audit.get("decision", {}).get("label") or "strict_replay_mismatch")
elif writeback.get("status") == "algorithmic_no_acceptance":
    label = "algorithmic_no_acceptance"
elif writeback.get("status") == "failed_invariant":
    label = "resize_forward_or_gradient_mismatch"
elif all(checks.values()) and run_rc == 0:
    label = "repair_pass"
payload = {
    "schema": "tausb.p1-det-resize-gate-summary.v1",
    "label": label,
    "checks": checks,
    "stage_exit_code": run_rc,
    "pass": label == "repair_pass",
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(label)
PY
)"
summary_rc=$?
set -e
if [[ "${summary_rc}" -ne 0 ]]; then
  gate_label="infra_failure"
fi
if [[ "${gate_label}" != "repair_pass" ]]; then
  run_rc=22
fi

if [[ -d "${ARTIFACT_ROOT}" ]]; then
  find "${ARTIFACT_ROOT}" -type f ! -name manifest.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "${ARTIFACT_ROOT}/manifest.sha256"
  artifact_bytes="$(du -sb "${ARTIFACT_ROOT}" | awk '{print $1}')"
  if [[ "${artifact_bytes}" -gt "${MAX_ARTIFACT_BYTES}" ]]; then
    run_rc=21
    gate_label="infra_failure"
  fi
fi
total_elapsed="$(( $(date +%s) - STARTED_SECONDS ))"
if [[ "${total_elapsed}" -gt "${HARD_CAP_SECONDS}" ]]; then
  run_rc=124
  gate_label="performance_gate_failed"
fi

status="completed"
if [[ "${run_rc}" -ne 0 ]]; then
  status="failed"
fi
printf '{"schema":"tausb.p1-det-resize-controller.v1","status":"%s","exit_code":%d,"label":"%s","execution_commit":"%s","config_sha256":"%s","elapsed_seconds":%d,"hard_cap_seconds":480}\n' \
  "${status}" "${run_rc}" "${gate_label}" "${EXPECTED_COMMIT}" \
  "${EXPECTED_CONFIG_SHA256}" "${total_elapsed}" \
  > "${CONTROL_ROOT}/controller_status.json"
exit "${run_rc}"
