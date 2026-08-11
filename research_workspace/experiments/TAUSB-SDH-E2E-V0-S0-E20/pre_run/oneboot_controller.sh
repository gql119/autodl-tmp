#!/usr/bin/env bash
set -Eeuo pipefail

: "${REPOSITORY_ROOT:?REPOSITORY_ROOT must identify the reviewed clean checkout}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT must identify the reviewed execution commit}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SHUTDOWN_BIN="${SHUTDOWN_BIN:-/usr/bin/shutdown}"
DEVICE="${DEVICE:-0}"

shutdown_once() {
  local rc=$?
  trap - EXIT INT TERM
  echo "[OneBoot] controller_exit=${rc}; requesting AutoDL shutdown"
  sync || true
  "${SHUTDOWN_BIN}" -h now || true
  exit "${rc}"
}

trap shutdown_once EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${REPOSITORY_ROOT}/ue_project"
"${PYTHON_BIN}" -u -m ue_framework.tools.run_tausb_sdh_e2e_v0_oneboot \
  --repository-root "${REPOSITORY_ROOT}" \
  --expected-commit "${EXPECTED_COMMIT}" \
  --python-bin "${PYTHON_BIN}" \
  --device "${DEVICE}"
