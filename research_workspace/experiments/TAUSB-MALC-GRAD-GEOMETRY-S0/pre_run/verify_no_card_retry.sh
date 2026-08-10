#!/usr/bin/env bash
set -Eeuo pipefail

CODE_COMMIT="18304b96c45360cfba5168d97d21d2961a13f390"
PROJECT_ROOT="/root/tausb-malc-geometry-wt-v2/ue_project"
FORMAL_ROOT="/root/tausb-sirc-runs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1/geometry"
OLD_CONTROL="/root/tausb-sirc-runs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1/control/geometry-seed0-18304b96"
R1_CONTROL="/root/tausb-sirc-runs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1/control/geometry-seed0-18304b96-r1"
R1_SESSION="tausb-malc-geometry-s0-r1"
WRAPPER="/root/run_tausb_malc_geometry_cost_guard.sh"
WRAPPER_SHA256="06fd902397867482cbdb0fc12a9261455be06e8c5dd0b1dd9724be4f2dc8187d"
PYTHON_BIN="/root/miniconda3/bin/python"

echo "HOSTNAME=$(hostname)"
cd "${PROJECT_ROOT}"
test "$(git rev-parse HEAD)" = "${CODE_COMMIT}"
test "$(git ls-files ue_framework configs | wc -l | tr -d ' ')" = "131"
git diff-index --quiet HEAD -- ue_framework configs
test -z "$(git diff --name-only -- ue_framework configs)"
test -z "$(git diff --cached --name-only -- ue_framework configs)"
unexpected_untracked="$(
  git ls-files --others --exclude-standard -- ue_framework configs \
    | grep -Ev '(^|/)__pycache__/.*\.pyc$' || true
)"
test -z "${unexpected_untracked}"
echo "SOURCE_SCOPE=clean_except_python_cache"

test -f "${OLD_CONTROL}/cost-guard-status.json"
test -f "${OLD_CONTROL}/geometry-seed0.log"
grep -Fq '"state":"failed"' "${OLD_CONTROL}/cost-guard-status.json"
grep -Fq "KeyError: 'semantic_bank_sha256'" "${OLD_CONTROL}/geometry-seed0.log"
echo "OLD_FAILURE=preserved_and_confirmed"

test ! -e "${FORMAL_ROOT}"
test ! -e "${R1_CONTROL}"
if tmux has-session -t "${R1_SESSION}" 2>/dev/null; then
  echo "R1_SESSION=exists"
  exit 51
fi
echo "FORMAL_ROOT=fresh"
echo "R1_CONTROL=fresh"
echo "R1_SESSION=absent"

bash -n "${WRAPPER}"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = "${WRAPPER_SHA256}"
grep -Fq 'prior["semantic_bank_hash"]' "${WRAPPER}"
grep -Fq 'prior["c2lm_basis_hash"]' "${WRAPPER}"
if grep -Fq 'prior["semantic_bank_sha256"]' "${WRAPPER}"; then
  echo "WRAPPER_SCHEMA=stale_semantic_key"
  exit 52
fi
if grep -Fq 'prior["c2lm_basis_sha256"]' "${WRAPPER}"; then
  echo "WRAPPER_SCHEMA=stale_c2lm_key"
  exit 53
fi
grep -Fq 'config_resolved.json' "${WRAPPER}"
echo "WRAPPER_SHA256=${WRAPPER_SHA256}"
echo "WRAPPER_SCHEMA=corrected"

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

import torch

current = json.loads(
    Path("/root/tausb-malc-geometry-input-audit.json").read_text(encoding="utf-8")
)
prior = json.loads(
    Path(
        "/root/tausb-sirc-runs/TAUSB-SIRC-MALC-CGR-MAP50-v2/"
        "mechanism/input_audit.json"
    ).read_text(encoding="utf-8")
)
assert current["pass"] is True
assert current["artifact_root_fresh"] is True
assert current["calibration_images"] == 64
assert current["heldout_images"] == 96
assert prior["semantic_bank_hash"] == "0b8a94efc55155bea20a1ec799bfac14c8a6f11fd6530538f3e0437b37c0dd4b"
assert prior["c2lm_basis_hash"] == "8350c0a608150839c98a8dad8db862d0c9dfaeca4714f05d1714afac0f30cfa5"
assert torch.cuda.is_available() is False
assert torch.cuda.device_count() == 0
print("ACTUAL_REMOTE_JSON_ASSERTION=pass")
print("CUDA_AVAILABLE=", torch.cuda.is_available())
print("CUDA_COUNT=", torch.cuda.device_count())
PY

echo "NO_CARD_RETRY_GATE=pass"
