#!/usr/bin/env bash
set -Eeuo pipefail

: "${REPOSITORY_ROOT:?REPOSITORY_ROOT must identify the reviewed clean checkout}"
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

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SHUTDOWN_BIN="${SHUTDOWN_BIN:-/usr/bin/shutdown}"
DEVICE="${DEVICE:-0}"

shutdown_once() {
  local rc=$?
  trap - EXIT INT TERM
  echo "[SparseE20] controller_exit=${rc}; requesting AutoDL shutdown"
  sync || true
  "${SHUTDOWN_BIN}" -h now || true
  exit "${rc}"
}

trap shutdown_once EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${REPOSITORY_ROOT}/ue_project"
"${PYTHON_BIN}" -u -m ue_framework.tools.run_tausb_sdh_sparse_e20 \
  --repository-root "${REPOSITORY_ROOT}" \
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
  --comparison-root "${COMPARISON_ROOT}"
