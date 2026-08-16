#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SHUTDOWN_BIN="${SHUTDOWN_BIN:-/usr/bin/shutdown}"

shutdown_once() {
  local rc=$?
  trap - EXIT INT TERM
  if mountpoint -q "${REQUIRED_STORAGE_ROOT:-/missing}"; then
    mkdir -p "${CONTROL_ROOT:-${REQUIRED_STORAGE_ROOT}/tausb-dgcaip-control/fallback}"
    printf '{"schema":"tausb.dgcaip-d0-wrapper-terminal.v1","exit_code":%d,"shutdown_requested":true}\n' "${rc}" \
      > "${CONTROL_ROOT:-${REQUIRED_STORAGE_ROOT}/tausb-dgcaip-control/fallback}/wrapper_terminal.json"
  fi
  echo "[DGCAIP-D0] controller_exit=${rc}; requesting AutoDL shutdown"
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
: "${CONFIG_PATH:?reviewed D0 config is required}"
: "${ARTIFACT_ROOT:?unique D0 artifact root is required}"
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
    *) echo "[DGCAIP-D0] path outside data disk: ${growing_path}" >&2; exit 20 ;;
  esac
done

test "$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${REPOSITORY_ROOT}" status --porcelain --untracked-files=all)"
test -f "${CONFIG_PATH}"
test ! -e "${ARTIFACT_ROOT}"
test ! -e "${CONTROL_ROOT}"

mkdir -p "${CONTROL_ROOT}" "${CACHE_ROOT}/xdg" "${CACHE_ROOT}/torch" \
  "${CACHE_ROOT}/yolo" "${TMP_ROOT}"
export TMPDIR="${TMP_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TORCH_HOME="${CACHE_ROOT}/torch"
export YOLO_CONFIG_DIR="${CACHE_ROOT}/yolo"

printf '{"schema":"tausb.dgcaip-d0-controller.v1","status":"running","execution_commit":"%s"}\n' \
  "${EXPECTED_COMMIT}" > "${CONTROL_ROOT}/controller_status.json"

cd "${REPOSITORY_ROOT}/ue_project"
"${PYTHON_BIN}" -c \
  'import pathlib,sys,yaml; from ue_framework.methods.sdh_experiment import validate_sdh_experiment_config; p=pathlib.Path(sys.argv[1]); c=yaml.safe_load(p.read_text(encoding="utf-8")); validate_sdh_experiment_config(c); assert c["dgcaip"]["run_mode"] == "d0"; assert pathlib.Path(c["runtime"]["artifact_root"]) == pathlib.Path(sys.argv[2])' \
  "${CONFIG_PATH}" "${ARTIFACT_ROOT}"

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
printf '{"schema":"tausb.dgcaip-d0-controller.v1","status":"%s","exit_code":%d,"execution_commit":"%s"}\n' \
  "${status}" "${run_rc}" "${EXPECTED_COMMIT}" \
  > "${CONTROL_ROOT}/controller_status.json"
exit "${run_rc}"
