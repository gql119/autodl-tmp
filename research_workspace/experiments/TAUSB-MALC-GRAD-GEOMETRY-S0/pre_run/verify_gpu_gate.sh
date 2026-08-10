#!/usr/bin/env bash
set -Eeuo pipefail

CODE_COMMIT="18304b96c45360cfba5168d97d21d2961a13f390"
PROJECT_ROOT="/root/tausb-malc-geometry-wt-v2/ue_project"
ARTIFACT_ROOT="/root/tausb-sirc-runs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1/geometry"
CONTROL_ROOT="/root/tausb-sirc-runs/TAUSB-MALC-GRAD-GEOMETRY-AUDIT-v1/control/geometry-seed0-18304b96-r1"
WRAPPER="/root/run_tausb_malc_geometry_cost_guard.sh"
WRAPPER_SHA256="06fd902397867482cbdb0fc12a9261455be06e8c5dd0b1dd9724be4f2dc8187d"
SESSION="tausb-malc-geometry-s0-r1"
PYTHON_BIN="/root/miniconda3/bin/python"

echo "HOSTNAME=$(hostname)"
cd "${PROJECT_ROOT}"
test "$(git rev-parse HEAD)" = "${CODE_COMMIT}"
echo "HEAD=${CODE_COMMIT}"
test "$(git ls-files ue_framework configs | wc -l | tr -d ' ')" = "131"
git diff-index --quiet HEAD -- ue_framework configs
test -z "$(git diff --name-only -- ue_framework configs)"
test -z "$(git diff --cached --name-only -- ue_framework configs)"
unexpected_untracked="$(
  git ls-files --others --exclude-standard -- ue_framework configs \
    | grep -Ev '(^|/)__pycache__/.*\.pyc$' || true
)"
test -z "${unexpected_untracked}"
echo "SOURCE_SCOPE=clean"

bash -n "${WRAPPER}"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = "${WRAPPER_SHA256}"
echo "WRAPPER_SHA256=${WRAPPER_SHA256}"
test ! -e "${ARTIFACT_ROOT}"
test ! -e "${CONTROL_ROOT}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "SESSION=exists"
  exit 41
fi
echo "FORMAL_ROOT=fresh"
echo "CONTROL_ROOT=fresh"
echo "SESSION=absent"
test -x /usr/bin/shutdown
echo "SHUTDOWN=executable"

disk_free="$(df -B1 --output=avail /root | tail -n 1 | tr -d ' ')"
test "${disk_free}" -ge 10737418240
echo "DISK_FREE_BYTES=${disk_free}"

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
test "${gpu_count}" = "1"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
case "${gpu_name}" in
  *4090*) ;;
  *) echo "Unexpected GPU: ${gpu_name}"; exit 42 ;;
esac
compute_apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
test -z "${compute_apps}"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
echo "COMPUTE_APPS=none"

"${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

import torch
import ultralytics
import yaml

from ue_framework.methods.sirc_malc_geometry import validate_geometry_config
from ue_framework.methods.sirc_probe import validate_sirc_config

config_path = Path(
    "ue_framework/configs/exp_voc_person_malc_grad_geometry_audit_v1.yaml"
)
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
validate_geometry_config(config)
validate_sirc_config(config)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert "4090" in torch.cuda.get_device_name(0)

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
assert prior["semantic_bank_hash"] == config["spec"]["semantic_bank_sha256"]
assert prior["c2lm_basis_hash"] == config["spec"]["c2lm_basis_sha256"]
print("PYTHON=", __import__("sys").version.split()[0])
print("TORCH=", torch.__version__)
print("ULTRALYTICS=", ultralytics.__version__)
print("CUDA_AVAILABLE=", torch.cuda.is_available())
print("CUDA_COUNT=", torch.cuda.device_count())
print("CUDA_NAME=", torch.cuda.get_device_name(0))
print("CONFIG_AND_INPUTS=pass")
PY

echo "GPU_GATE=pass"
