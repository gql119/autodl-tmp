#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SHUTDOWN_BIN="${SHUTDOWN_BIN:-/usr/bin/shutdown}"
HARD_CAP_SECONDS=300
MAX_ARTIFACT_BYTES=104857600
STARTED_SECONDS="$(date +%s)"
CONTROL_ROOT_VALIDATED=0

shutdown_once() {
  local rc=$?
  trap - EXIT INT TERM
  if mountpoint -q "${REQUIRED_STORAGE_ROOT:-/missing}"; then
    local terminal_root="${REQUIRED_STORAGE_ROOT}/tausb-dgcaip-control/fallback"
    if [[ "${CONTROL_ROOT_VALIDATED}" -eq 1 ]]; then
      terminal_root="${CONTROL_ROOT}"
    fi
    mkdir -p "${terminal_root}"
    local terminal_path="${terminal_root}/wrapper_terminal.json"
    if [[ -e "${terminal_path}" ]]; then
      terminal_path="$(mktemp "${terminal_root}/wrapper_terminal.XXXXXX.json")"
    fi
    printf '{"schema":"tausb.p1-determinism-wrapper-terminal.v1","exit_code":%d,"shutdown_requested":true}\n' \
      "${rc}" > "${terminal_path}"
    printf '{"schema":"tausb.p1-determinism-controller-terminal.v1","exit_code":%d,"hard_cap_seconds":300}\n' \
      "${rc}" > "${terminal_root}/controller_terminal.json"
  fi
  echo "[P1-DET] controller_exit=${rc}; requesting AutoDL shutdown"
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
: "${CONFIG_PATH:?reviewed P1 determinism config is required}"
: "${ARTIFACT_ROOT:?unique P1 determinism artifact root is required}"
: "${CONTROL_ROOT:?control root is required}"
: "${CACHE_ROOT:?cache root is required}"
: "${TMP_ROOT:?temporary root is required}"

command -v timeout >/dev/null 2>&1
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
    *) echo "[P1-DET] path outside data disk: ${growing_path}" >&2; exit 20 ;;
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

printf '{"schema":"tausb.p1-determinism-controller.v1","status":"running","execution_commit":"%s","config_sha256":"%s","hard_cap_seconds":300}\n' \
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
assert config["spec"]["spec_id"] == "TAUSB-SDH-DGCAIP-P1-DETERMINISM-AUDIT-v1"
assert config["dgcaip"]["run_mode"] == "p1_determinism_audit"
assert config["audit"]["normal_lanes"] == ["shared", "reset", "fresh"]
assert config["audit"]["strict_lanes"] == ["fresh"]
assert config["audit"]["zero_parameter_updates"] is True
assert config["audit"]["total_hard_cap_seconds"] == 300
assert config["mechanism"]["optimization_steps"] == 1
assert config["mechanism"]["max_seconds"] == 300
assert pathlib.Path(config["runtime"]["artifact_root"]) == artifact_root
PY

run_lane() {
  local mode="$1"
  local elapsed remaining
  elapsed="$(( $(date +%s) - STARTED_SECONDS ))"
  remaining="$(( HARD_CAP_SECONDS - elapsed ))"
  if [[ "${remaining}" -le 0 ]]; then
    echo "[P1-DET] hard cap exhausted before ${mode}" >&2
    return 124
  fi
  if [[ "${mode}" == "strict" ]]; then
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      timeout --signal=TERM --kill-after=15s "${remaining}s" \
      "${PYTHON_BIN}" -u -m ue_framework.tools.run_p1_determinism_audit \
        --config "${CONFIG_PATH}" --mode "${mode}"
  else
    env -u CUBLAS_WORKSPACE_CONFIG \
      timeout --signal=TERM --kill-after=15s "${remaining}s" \
        "${PYTHON_BIN}" -u -m ue_framework.tools.run_p1_determinism_audit \
          --config "${CONFIG_PATH}" --mode "${mode}"
  fi
}

set +e
run_lane normal
run_rc=$?
if [[ "${run_rc}" -eq 0 ]]; then
  run_lane strict
  run_rc=$?
fi
if [[ "${run_rc}" -eq 0 ]]; then
  run_lane summarize
  run_rc=$?
fi
set -e

if [[ "${run_rc}" -eq 0 ]]; then
  find "${ARTIFACT_ROOT}" -maxdepth 1 -type f ! -name manifest.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > "${ARTIFACT_ROOT}/manifest.sha256"
fi

if [[ -d "${ARTIFACT_ROOT}" ]]; then
  artifact_bytes="$(du -sb "${ARTIFACT_ROOT}" | awk '{print $1}')"
  if [[ "${artifact_bytes}" -gt "${MAX_ARTIFACT_BYTES}" ]]; then
    echo "[P1-DET] artifact cap exceeded: ${artifact_bytes}" >&2
    run_rc=21
  fi
fi
total_elapsed="$(( $(date +%s) - STARTED_SECONDS ))"
if [[ "${total_elapsed}" -gt "${HARD_CAP_SECONDS}" ]]; then
  echo "[P1-DET] total wall clock exceeded hard cap: ${total_elapsed}" >&2
  run_rc=124
fi

status="completed"
if [[ "${run_rc}" -ne 0 ]]; then
  status="failed"
fi
printf '{"schema":"tausb.p1-determinism-controller.v1","status":"%s","exit_code":%d,"execution_commit":"%s","config_sha256":"%s","hard_cap_seconds":300}\n' \
  "${status}" "${run_rc}" "${EXPECTED_COMMIT}" "${EXPECTED_CONFIG_SHA256}" \
  > "${CONTROL_ROOT}/controller_status.json"
exit "${run_rc}"
