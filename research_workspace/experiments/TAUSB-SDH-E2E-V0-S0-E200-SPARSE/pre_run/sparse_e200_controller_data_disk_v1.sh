#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SHUTDOWN_BIN="${SHUTDOWN_BIN:-/usr/bin/shutdown}"
DEVICE="${DEVICE:-0}"

shutdown_once() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ -n "${REQUIRED_STORAGE_ROOT:-}" ]] \
    && [[ -n "${LOG_ROOT:-}" ]] \
    && mountpoint -q "${REQUIRED_STORAGE_ROOT}"; then
    mkdir -p "${LOG_ROOT}"
    local terminal_path="${LOG_ROOT}/wrapper_terminal.json"
    if [[ -e "${terminal_path}" ]]; then
      terminal_path="$(mktemp "${LOG_ROOT}/wrapper_terminal.XXXXXX.json")"
    fi
    printf '{"schema":"tausb.sdh-e200-wrapper-terminal.v1","exit_code":%d,"shutdown_requested":true}\n' "${rc}" \
      > "${terminal_path}"
  fi
  echo "[SparseE200] controller_exit=${rc}; evidence flushed; requesting AutoDL shutdown"
  sync || true
  "${SHUTDOWN_BIN}" -h now || true
  exit "${rc}"
}

trap shutdown_once EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

: "${REPOSITORY_ROOT:?REPOSITORY_ROOT must identify the reviewed clean checkout}"
: "${REQUIRED_STORAGE_ROOT:?REQUIRED_STORAGE_ROOT must identify the mounted AutoDL data disk}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must identify the reviewed execution commit}"
: "${MECHANISM_ROOT:?MECHANISM_ROOT is required}"
: "${MECHANISM_CONFIG:?MECHANISM_CONFIG is required}"
: "${BASE_CONFIG:?BASE_CONFIG is required}"
: "${DATASET_ROOT:?DATASET_ROOT is required}"
: "${BINDING_ROOT:?BINDING_ROOT is required}"
: "${RUN_ROOT_PREFIX:?RUN_ROOT_PREFIX is required}"
: "${CONTROL_ROOT:?CONTROL_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${COMPARISON_ROOT:?COMPARISON_ROOT is required}"
: "${CACHE_ROOT:?CACHE_ROOT is required}"
: "${TMP_ROOT:?TMP_ROOT is required}"

command -v timeout >/dev/null 2>&1 || {
  echo "[SparseE200] GNU timeout is required for the 9-hour outer hard cap" >&2
  exit 20
}

REQUIRED_STORAGE_ROOT="$(realpath -m "${REQUIRED_STORAGE_ROOT}")"
REPOSITORY_ROOT="$(realpath -m "${REPOSITORY_ROOT}")"
DATASET_ROOT="$(realpath -m "${DATASET_ROOT}")"
BINDING_ROOT="$(realpath -m "${BINDING_ROOT}")"
RUN_ROOT_PREFIX="$(realpath -m "${RUN_ROOT_PREFIX}")"
CONTROL_ROOT="$(realpath -m "${CONTROL_ROOT}")"
LOG_ROOT="$(realpath -m "${LOG_ROOT}")"
COMPARISON_ROOT="$(realpath -m "${COMPARISON_ROOT}")"
CACHE_ROOT="$(realpath -m "${CACHE_ROOT}")"
TMP_ROOT="$(realpath -m "${TMP_ROOT}")"

export TMPDIR="${TMP_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export YOLO_CONFIG_DIR="${CACHE_ROOT}/yolo"

if ! mountpoint -q "${REQUIRED_STORAGE_ROOT}"; then
  echo "[SparseE200] required storage root is not a mountpoint: ${REQUIRED_STORAGE_ROOT}" >&2
  exit 20
fi

for growing_path in \
  "${REPOSITORY_ROOT}" "${DATASET_ROOT}" "${BINDING_ROOT}" \
  "${RUN_ROOT_PREFIX}" "${CONTROL_ROOT}" "${LOG_ROOT}" \
  "${COMPARISON_ROOT}" "${CACHE_ROOT}" "${TMP_ROOT}"; do
  case "${growing_path}" in
    "${REQUIRED_STORAGE_ROOT}"|"${REQUIRED_STORAGE_ROOT}"/*) ;;
    *) echo "[SparseE200] path outside required storage root: ${growing_path}" >&2; exit 20 ;;
  esac
done
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${YOLO_CONFIG_DIR}"

cd "${REPOSITORY_ROOT}/ue_project"
timeout --signal=TERM --kill-after=60s 32400s \
"${PYTHON_BIN}" -u -m ue_framework.tools.run_tausb_sdh_sparse_e20 \
  --repository-root "${REPOSITORY_ROOT}" \
  --required-storage-root "${REQUIRED_STORAGE_ROOT}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --python-bin "${PYTHON_BIN}" \
  --device "${DEVICE}" \
  --mechanism-root "${MECHANISM_ROOT}" \
  --mechanism-config "${MECHANISM_CONFIG}" \
  --base-config "${BASE_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --binding-root "${BINDING_ROOT}" \
  --run-root-prefix "${RUN_ROOT_PREFIX}" \
  --control-root "${CONTROL_ROOT}" \
  --log-root "${LOG_ROOT}" \
  --comparison-root "${COMPARISON_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --tmp-root "${TMP_ROOT}" \
  --victim-epochs 200
