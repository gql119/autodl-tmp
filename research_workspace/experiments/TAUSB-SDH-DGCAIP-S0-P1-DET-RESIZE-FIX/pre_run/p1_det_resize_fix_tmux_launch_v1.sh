#!/usr/bin/env bash
set -Eeuo pipefail

: "${REPOSITORY_ROOT:?reviewed detached checkout is required}"
: "${REQUIRED_STORAGE_ROOT:?AutoDL data-disk root is required}"
: "${EXPECTED_COMMIT:?reviewed commit is required}"
: "${EXPECTED_CONFIG_SHA256:?reviewed config hash is required}"
: "${CONFIG_PATH:?reviewed config is required}"
: "${ARTIFACT_ROOT:?unique artifact root is required}"
: "${CONTROL_ROOT:?unique control root is required}"
: "${CACHE_ROOT:?cache root is required}"
: "${TMP_ROOT:?temporary root is required}"
: "${CONTROLLER_PATH:?reviewed controller is required}"

SESSION_NAME="tausb-p1-det-resize-fix"
HARD_CAP_SECONDS=480
SHUTDOWN_BIN="${SHUTDOWN_BIN:-/usr/bin/shutdown}"

shutdown_on_launch_failure() {
  local rc=$?
  trap - EXIT
  if [[ "${rc}" -ne 0 ]]; then
    echo "[P1-DET-RESIZE] launcher_exit=${rc}; requesting AutoDL shutdown" >&2
    "${SHUTDOWN_BIN}" -h now || true
  fi
  exit "${rc}"
}

trap shutdown_on_launch_failure EXIT
command -v tmux >/dev/null 2>&1
command -v timeout >/dev/null 2>&1
test -f "${CONTROLLER_PATH}"
test ! -e "${ARTIFACT_ROOT}"
test ! -e "${CONTROL_ROOT}"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "[P1-DET-RESIZE] session already exists: ${SESSION_NAME}" >&2
  exit 30
fi
mkdir -p "$(dirname "${CONTROL_ROOT}")"

tmux new-session -d -s "${SESSION_NAME}" \
  "env REPOSITORY_ROOT='${REPOSITORY_ROOT}' REQUIRED_STORAGE_ROOT='${REQUIRED_STORAGE_ROOT}' EXPECTED_COMMIT='${EXPECTED_COMMIT}' EXPECTED_CONFIG_SHA256='${EXPECTED_CONFIG_SHA256}' CONFIG_PATH='${CONFIG_PATH}' ARTIFACT_ROOT='${ARTIFACT_ROOT}' CONTROL_ROOT='${CONTROL_ROOT}' CACHE_ROOT='${CACHE_ROOT}' TMP_ROOT='${TMP_ROOT}' timeout --signal=TERM --kill-after=15s '${HARD_CAP_SECONDS}s' bash '${CONTROLLER_PATH}' > '${CONTROL_ROOT}.outer.log' 2>&1"
trap - EXIT
echo "[P1-DET-RESIZE] launched ${SESSION_NAME}"
