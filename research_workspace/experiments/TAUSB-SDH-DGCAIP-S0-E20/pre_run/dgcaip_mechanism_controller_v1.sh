#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SHUTDOWN_BIN="${SHUTDOWN_BIN:-/usr/bin/shutdown}"

shutdown_once() {
  local rc=$?
  trap - EXIT INT TERM
  if mountpoint -q "${REQUIRED_STORAGE_ROOT:-/missing}"; then
    mkdir -p "${CONTROL_ROOT:-${REQUIRED_STORAGE_ROOT}/tausb-dgcaip-control/fallback}"
    local terminal_path="${CONTROL_ROOT:-${REQUIRED_STORAGE_ROOT}/tausb-dgcaip-control/fallback}/wrapper_terminal.json"
    if [[ -e "${terminal_path}" ]]; then
      terminal_path="$(mktemp "${CONTROL_ROOT:-${REQUIRED_STORAGE_ROOT}/tausb-dgcaip-control/fallback}/wrapper_terminal.XXXXXX.json")"
    fi
    printf '{"schema":"tausb.dgcaip-mechanism-wrapper-terminal.v1","exit_code":%d,"shutdown_requested":true}\n' "${rc}" \
      > "${terminal_path}"
  fi
  echo "[DGCAIP-Mechanism] controller_exit=${rc}; requesting AutoDL shutdown"
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
: "${EXPECTED_CONFIG_SHA256:?reviewed bound-config hash is required}"
: "${CONFIG_PATH:?reviewed mechanism config is required}"
: "${ARTIFACT_ROOT:?unique mechanism artifact root is required}"
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
    *) echo "[DGCAIP-Mechanism] path outside data disk: ${growing_path}" >&2; exit 20 ;;
  esac
done

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

printf '{"schema":"tausb.dgcaip-mechanism-controller.v1","status":"running","execution_commit":"%s","config_sha256":"%s"}\n' \
  "${EXPECTED_COMMIT}" "${EXPECTED_CONFIG_SHA256}" \
  > "${CONTROL_ROOT}/controller_status.json"

cd "${REPOSITORY_ROOT}/ue_project"
"${PYTHON_BIN}" - "${CONFIG_PATH}" "${ARTIFACT_ROOT}" <<'PY'
import hashlib
import pathlib
import sys

import yaml

from ue_framework.methods.sdh_experiment import validate_sdh_experiment_config


config_path = pathlib.Path(sys.argv[1])
artifact_root = pathlib.Path(sys.argv[2])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
validate_sdh_experiment_config(config)
assert config["dgcaip"]["run_mode"] == "mechanism"
assert pathlib.Path(config["runtime"]["artifact_root"]) == artifact_root
for path_key, hash_key in (
    ("d0_report", "d0_report_sha256"),
    ("source_p1_state", "source_p1_state_sha256"),
    ("source_p1_metrics", "source_p1_metrics_sha256"),
):
    source = pathlib.Path(config["dgcaip"][path_key])
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual == str(config["dgcaip"][hash_key]).lower(), source
PY

set +e
timeout --signal=TERM --kill-after=30s 1200s \
  "${PYTHON_BIN}" -u -m ue_framework.tools.run_tausb_sdh \
    --config "${CONFIG_PATH}" \
    --stage mechanism
run_rc=$?
set -e

status="completed"
if [[ "${run_rc}" -ne 0 ]]; then
  status="failed"
fi
printf '{"schema":"tausb.dgcaip-mechanism-controller.v1","status":"%s","exit_code":%d,"execution_commit":"%s","config_sha256":"%s"}\n' \
  "${status}" "${run_rc}" "${EXPECTED_COMMIT}" "${EXPECTED_CONFIG_SHA256}" \
  > "${CONTROL_ROOT}/controller_status.json"
exit "${run_rc}"
