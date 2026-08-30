#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 <expected-commit>" >&2
  exit 2
fi

expected_commit="$1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/../../../.." && pwd)"
data_root="/root/autodl-tmp"
python_bin="/root/miniconda3/bin/python"
exp_root="${data_root}/tausb-dgcaip-runs/TAUSB-SDH-DGCAIP-S0-P4-SPARSE-E20-R2"
wrapper_log_root="${data_root}/tausb-dgcaip-wrapper-logs"

shutdown_instance() {
  /usr/bin/shutdown -h now || true
}
trap shutdown_instance EXIT INT TERM HUP

mkdir -p \
  "${data_root}/tausb-cache/xdg/torch/kernels" \
  "${data_root}/tausb-cache/torch" \
  "${data_root}/tausb-cache/yolo" \
  "${data_root}/tausb-tmp" \
  "${wrapper_log_root}"
cd "${repository_root}/ue_project"

/usr/bin/timeout --signal=TERM --kill-after=60s 7200s \
  "${python_bin}" -u -m ue_framework.tools.run_tausb_sdh_dgcaip_p4_oneboot \
  --repository-root "${repository_root}" \
  --required-storage-root "${data_root}" \
  --expected-commit "${expected_commit}" \
  --python-bin "${python_bin}" \
  --device 0 \
  --mechanism-config "${repository_root}/ue_project/ue_framework/configs/tausb_sdh_dgcaip_p4_sparse_e20_v1.yaml" \
  --mechanism-root "${exp_root}" \
  --base-config "${repository_root}/ue_project/ue_framework/configs/exp_voc_person_sdh_lfc_cicr_cgr_nla_map50_v3.yaml" \
  --dataset-root "${data_root}/ue_project/VOC_0712_Kaggle_Ready" \
  --binding-root "${exp_root}-BINDING" \
  --run-root-prefix "${exp_root}-VICTIM" \
  --control-root "${exp_root}-CONTROL" \
  --sparse-control-root "${exp_root}-SPARSE-CONTROL" \
  --log-root "${exp_root}-LOGS" \
  --comparison-root "${exp_root}-COMPARISON" \
  --cache-root "${data_root}/tausb-cache" \
  --tmp-root "${data_root}/tausb-tmp" \
  >"${wrapper_log_root}/TAUSB-SDH-DGCAIP-S0-P4-SPARSE-E20-R2.outer.log" 2>&1
