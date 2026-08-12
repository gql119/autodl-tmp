#!/usr/bin/env bash
set -Eeuo pipefail

failure_shutdown() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ ${rc} -ne 0 ]]; then
    echo "[LaunchGate-R4] failed exit=${rc}; requesting AutoDL shutdown" >&2
    sync || true
    /usr/bin/shutdown -h now || true
  fi
  exit "${rc}"
}

trap failure_shutdown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

COMMIT=83cfb21c11195e1b1e034db3422716a34b18e166
CHECKOUT=/root/tausb-sdh-checkouts/e2e-v0-83cfb21-r4-worktree
WRAPPER=${CHECKOUT}/research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E20/pre_run/oneboot_r4_resume_controller.sh
WRAPPER_SHA256=778093577f45ddd31bc705f22d6b8d4bc14a944b44e96ade90b8110da417b7e1
TOOL_SHA256=91bb9d1db1c86bd1a4498ad5a018224972c407b0db1677f95a41ba85feefce61
MANIFEST_WRITER_SHA256=5ba9b8af6cd3bb04057b89e330773a1d054b0fc8138e22d87179bdc9ee0898c1
BINDING_ROOT=/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-BINDING-R4
BINDING_SHA256=f06efd60bf2adad91c0cfb148d8713747d9d3250e5d828f930d5bc2bc472c6b5
SELECTION_SHA256=46a73d262c14312af89e5c5496b0d13b42542845aba72861dd11e2863e6199b3
FROZEN_STATE=/root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-MECH/mechanism/p1_feasibility_sdh_state.pt
FROZEN_STATE_SHA256=c6c994384a563506126065382e35c941ba0bb0b2a21cd1d2dea63373bffd5168
SESSION=tausb-sdh-e2e-v0-oneboot-s0-r4

test -x /usr/bin/tmux
test -x /usr/bin/shutdown
test -x /root/miniconda3/bin/python
test "$(git -C "${CHECKOUT}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${CHECKOUT}" status --porcelain --untracked-files=no)"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = "${WRAPPER_SHA256}"
test "$(sha256sum "${CHECKOUT}/ue_project/ue_framework/tools/run_tausb_sdh_e2e_v0_oneboot.py" | awk '{print $1}')" = "${TOOL_SHA256}"
test "$(sha256sum "${CHECKOUT}/ue_project/ue_framework/stages/generate.py" | awk '{print $1}')" = "${MANIFEST_WRITER_SHA256}"
test "$(sha256sum "${BINDING_ROOT}/binding_report.json" | awk '{print $1}')" = "${BINDING_SHA256}"
test "$(sha256sum "${BINDING_ROOT}/smoke_train_selection.json" | awk '{print $1}')" = "${SELECTION_SHA256}"
test "$(sha256sum "${BINDING_ROOT}/smoke-c0.yaml" | awk '{print $1}')" = e225fcc8c4d9b0107b6d01c3e5f21e818285a50c9cc9374c84cb6febde98bbeb
test "$(sha256sum "${BINDING_ROOT}/smoke-m1.yaml" | awk '{print $1}')" = d0bf860b993be17794c12fd716c85e8430bf15e37c0417b8a41ebdf2ebb78d5f
test "$(sha256sum "${BINDING_ROOT}/e20-c0.yaml" | awk '{print $1}')" = a84c3f902b228055a291144653c0d5b8b42afa11aed9c3b72edf17e4dd303304
test "$(sha256sum "${BINDING_ROOT}/e20-m1.yaml" | awk '{print $1}')" = 0447fa992e95810a512a8616a76a5dcdb696e039459abc19c580e6cc46cefb38
test "$(sha256sum "${FROZEN_STATE}" | awk '{print $1}')" = "${FROZEN_STATE_SHA256}"

for root in \
  /root/tausb-sdh-control/TAUSB-SDH-E2E-V0-S0-E20-ONEBOOT-R4 \
  /root/tausb-sdh-logs/TAUSB-SDH-E2E-V0-S0-E20-ONEBOOT-R4 \
  /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-E20-COMPARISON-R4 \
  /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-R4-SMOKE-C0 \
  /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-R4-SMOKE-M1 \
  /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-R4-E20-C0 \
  /root/tausb-sdh-runs/TAUSB-SDH-E2E-V0-S0-R4-E20-M1
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
echo "[LaunchGate-R4] pass session=${SESSION} commit=${COMMIT} start=SMOKE_C0"
