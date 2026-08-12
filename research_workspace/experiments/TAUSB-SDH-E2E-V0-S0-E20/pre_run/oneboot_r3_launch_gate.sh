#!/usr/bin/env bash
set -Eeuo pipefail

failure_shutdown() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ ${rc} -ne 0 ]]; then
    echo "[LaunchGate-R3] failed exit=${rc}; requesting AutoDL shutdown" >&2
    sync || true
    /usr/bin/shutdown -h now || true
  fi
  exit "${rc}"
}

trap failure_shutdown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

COMMIT=34e28f1622f2b3f053de70e1cb0d013f62d42f15
CHECKOUT=/root/tausb-sdh-checkouts/e2e-v0-34e28f1-r3-worktree
WRAPPER=${CHECKOUT}/research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E20/pre_run/oneboot_r3_resume_controller.sh
WRAPPER_SHA256=92a7b53fc47e0f1f68dcb71ef016febe6f981ff9e0688535aab72c2d3c85889d
TOOL_SHA256=6d5461c124d3323509ad9c7d4256f4d900eb789940190f9ec60e3478a7b8e925
BINDING=/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-BINDING-R2/binding_report.json
BINDING_SHA256=a72615ddd66f2f679c4f1789d267e30cdbf1e33cd8595a32ebaec7ee10ddd029
FROZEN_STATE=/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-MECH/mechanism/p1_feasibility_sdh_state.pt
FROZEN_STATE_SHA256=c6c994384a563506126065382e35c941ba0bb0b2a21cd1d2dea63373bffd5168
SESSION=tausb-sdh-e2e-v0-oneboot-s0-r3

test -x /usr/bin/tmux
test -x /usr/bin/shutdown
test -x /root/miniconda3/bin/python
test "$(git -C "${CHECKOUT}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${CHECKOUT}" status --porcelain --untracked-files=no)"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = "${WRAPPER_SHA256}"
test "$(sha256sum "${CHECKOUT}/ue_project/ue_framework/tools/run_tausb_sdh_e2e_v0_oneboot.py" | awk '{print $1}')" = "${TOOL_SHA256}"
test "$(sha256sum "${BINDING}" | awk '{print $1}')" = "${BINDING_SHA256}"
test "$(sha256sum "${FROZEN_STATE}" | awk '{print $1}')" = "${FROZEN_STATE_SHA256}"

for root in \
  /root/tausb-sdh-control/TAUSB-SDH-E2E-V0-S0-E20-ONEBOOT-R3 \
  /root/tausb-sdh-logs/TAUSB-SDH-E2E-V0-S0-E20-ONEBOOT-R3 \
  /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-COMPARISON-R3 \
  /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-R2-SMOKE-C0 \
  /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-R2-SMOKE-M1 \
  /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-R2-E20-C0 \
  /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-R2-E20-M1
do
  test ! -e "${root}"
done
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  exit 41
fi

tmux new-session -d -s "${SESSION}" env \
  REPOSITORY_ROOT="${CHECKOUT}" \
  EXPECTED_COMMIT="${COMMIT}" \
  PYTHON_BIN=/root/miniconda3/bin/python \
  DEVICE=0 \
  bash "${WRAPPER}"
sleep 5
tmux has-session -t "${SESSION}"

trap - EXIT INT TERM
echo "[LaunchGate-R3] pass session=${SESSION} commit=${COMMIT} start=SMOKE_C0"
