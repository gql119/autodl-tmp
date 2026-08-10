#!/usr/bin/env bash
set -Eeuo pipefail

CODE_COMMIT="e3f674497569087a79dd3782fcdfbabd4e7c8d04"
REPOSITORY_ROOT="/root/tausb-sdh-checkouts/6f4d5d8-worktree"
PROJECT_ROOT="${REPOSITORY_ROOT}/ue_project"
ARTIFACT_ROOT="/root/tausb-sdh-runs/TAUSB-SDH-LFC-CICR-CGR-NLA-S0"
CONTROL_ROOT="/root/tausb-sdh-control/TAUSB-SDH-LFC-CICR-CGR-NLA-S0/hiding-e3f6744-r1"
INPUT_AUDIT="${REPOSITORY_ROOT}/research_workspace/experiments/TAUSB-SDH-LFC-CICR-CGR-NLA-S0/pre_run/remote_input_audit.json"
WRAPPER="/root/run_sdh_hiding_cost_guard_e3f6744.sh"
WRAPPER_SHA256="99fe8cbcac2a82d8af20b7df5f165688a1b090c586261233e3ec231e3c3f6419"
SESSION="tausb-sdh-hiding-s0-e3f6744-r1"
PYTHON_BIN="/root/miniconda3/bin/python"

failure_shutdown() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    echo "[LaunchGate] pre-run gate failed with exit=${rc}; requesting AutoDL shutdown" >&2
    sync || true
    /usr/bin/shutdown || true
  fi
  exit "${rc}"
}
trap failure_shutdown EXIT

cd "${PROJECT_ROOT}"
test "$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD)" = "${CODE_COMMIT}"
git -C "${REPOSITORY_ROOT}" diff-index --quiet HEAD -- ue_project/ue_framework research_workspace/sources
test -z "$(git -C "${REPOSITORY_ROOT}" diff --name-only -- ue_project/ue_framework research_workspace/sources)"
test -z "$(git -C "${REPOSITORY_ROOT}" diff --cached --name-only -- ue_project/ue_framework research_workspace/sources)"
echo "CODE_SNAPSHOT=clean:${CODE_COMMIT}"

test -x "${PYTHON_BIN}"
test -x /usr/bin/timeout
test -x /usr/bin/shutdown
test -x /usr/bin/tmux
test -f "${WRAPPER}"
test -f "${INPUT_AUDIT}"
bash -n "${WRAPPER}"
test "$(sha256sum "${WRAPPER}" | awk '{print $1}')" = "${WRAPPER_SHA256}"
echo "WRAPPER_SHA256=${WRAPPER_SHA256}"

test ! -e "${ARTIFACT_ROOT}"
test ! -e "${CONTROL_ROOT}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "SESSION_ALREADY_EXISTS=${SESSION}" >&2
  exit 41
fi
echo "FORMAL_ROOT=fresh"
echo "CONTROL_ROOT=fresh"
echo "SESSION=absent:${SESSION}"

disk_free="$(df -B1 --output=avail /root | tail -n 1 | tr -d ' ')"
test "${disk_free}" -ge 10737418240
echo "DISK_FREE_BYTES=${disk_free}"

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
test "${gpu_count}" -ge 1
compute_apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
test -z "${compute_apps}"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
echo "COMPUTE_APPS=none"

export PYTHONDONTWRITEBYTECODE=1
"${PYTHON_BIN}" - <<'PY'
import hashlib
import json
from pathlib import Path

import torch
import ultralytics
import yaml

from ue_framework.methods.sdh_experiment import (
    _canonical_json_sha256,
    _load_secret_bank,
    validate_sdh_experiment_config,
)

project = Path.cwd()
repository = project.parent
config = yaml.safe_load(
    Path("ue_framework/configs/tausb_sdh_mechanism_v3.yaml").read_text(
        encoding="utf-8"
    )
)
validate_sdh_experiment_config(config)
assert config["spec"]["seed"] == 0
assert config["dataset"]["target_class_id"] == 14
assert config["dataset"]["expected_train_images"] == 16551
assert config["dataset"]["expected_person_images"] == 6095
assert config["mechanism"]["eot_enabled"] is False
assert config["mechanism"]["jnd_enabled"] is False
assert config["hiding"]["max_seconds"] == 1200

dataset = Path(config["dataset"]["root"])
checkpoint = Path(config["model"]["surrogate_checkpoint"])
assert (dataset / "images/train").is_dir()
assert (dataset / "labels/train").is_dir()
assert checkpoint.is_file()
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
    "8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89"
)

audit = json.loads(
    (
        repository
        / "research_workspace/experiments/TAUSB-SDH-LFC-CICR-CGR-NLA-S0/pre_run/remote_input_audit.json"
    ).read_text(encoding="utf-8")
)
assert audit["pass"] is True
assert all(audit["checks"].values())
assert audit["dataset"]["train_label_content_manifest_sha256"] == (
    config["dataset"]["train_label_manifest_sha256"]
)

manifest = json.loads(
    (
        repository / "research_workspace/sources/secret_assets/manifest.json"
    ).read_text(encoding="utf-8")
)
assert _canonical_json_sha256(manifest) == config["secrets"]["manifest_sha256"]
secrets, primary_index, _ = _load_secret_bank(config, project)
assert tuple(secrets.shape) == (4, 3, 256, 256)
assert primary_index == 3

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() >= 1
print("PYTHON=" + __import__("sys").version.split()[0])
print("TORCH=" + torch.__version__)
print("ULTRALYTICS=" + ultralytics.__version__)
print("CUDA_NAME=" + torch.cuda.get_device_name(0))
print("CONFIG_INPUTS_CUDA=pass")
PY

tmux new-session -d -s "${SESSION}" /bin/bash "${WRAPPER}"
sleep 5
tmux has-session -t "${SESSION}"
trap - EXIT
echo "GPU_GATE=pass"
echo "SESSION=running:${SESSION}"
echo "LOG=${CONTROL_ROOT}/hiding.log"
