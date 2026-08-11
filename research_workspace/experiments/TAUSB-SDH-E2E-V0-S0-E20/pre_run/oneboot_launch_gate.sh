#!/usr/bin/env bash
set -Eeuo pipefail

failure_shutdown() {
  local rc=$?
  trap - EXIT INT TERM
  if [[ ${rc} -ne 0 ]]; then
    echo "[LaunchGate] failed exit=${rc}; requesting AutoDL shutdown" >&2
    sync || true
    /usr/bin/shutdown -h now || true
  fi
  exit "${rc}"
}

trap failure_shutdown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

SOURCE_REPO=/root/autodl-tmp
BRANCH=codex/tausb-sdh-lfc-cicr-cgr-nla-map50-v3
COMMIT=52d57fd005a318c912f5e43a5cf91dfe1357cddf
CHECKOUT=/root/tausb-sdh-checkouts/e2e-v0-52d57fd-worktree
WRAPPER=${CHECKOUT}/research_workspace/experiments/TAUSB-SDH-E2E-V0-S0-E20/pre_run/oneboot_controller.sh
WRAPPER_SHA256=4a18558f97bd4e5c6ab71b006069fbdd6ac8be922a0aa9486e467e714a14e345
TOOL_SHA256=3925d377c566c221bd752765a02fbfd67e32b338f0e3cd9e7b2b0c9613f0de6f
SESSION=tausb-sdh-e2e-v0-oneboot-s0-r1

test -x /usr/bin/tmux
test -x /usr/bin/shutdown
test -x /root/miniconda3/bin/python
test ! -e "${CHECKOUT}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  exit 41
fi

git -C "${SOURCE_REPO}" fetch origin "${BRANCH}"
git -C "${SOURCE_REPO}" cat-file -e "${COMMIT}^{commit}"
git -C "${SOURCE_REPO}" worktree add --detach "${CHECKOUT}" "${COMMIT}"
test "$(git -C "${CHECKOUT}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${CHECKOUT}" status --porcelain --untracked-files=no)"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = "${WRAPPER_SHA256}"
test "$(sha256sum "${CHECKOUT}/ue_project/ue_framework/tools/run_tausb_sdh_e2e_v0_oneboot.py" | awk '{print $1}')" = "${TOOL_SHA256}"

tmux new-session -d -s "${SESSION}" env \
  REPOSITORY_ROOT="${CHECKOUT}" \
  EXPECTED_COMMIT="${COMMIT}" \
  PYTHON_BIN=/root/miniconda3/bin/python \
  DEVICE=0 \
  bash "${WRAPPER}"
sleep 5
tmux has-session -t "${SESSION}"

trap - EXIT INT TERM
echo "[LaunchGate] pass session=${SESSION} commit=${COMMIT}"
