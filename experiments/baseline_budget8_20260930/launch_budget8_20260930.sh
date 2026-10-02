#!/usr/bin/env bash
set -uo pipefail
ROOT=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/baseline_dev_20260928/candidate_budget8_541e338/ue_project
BASE=/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/b2_adaptive_dev_20260926/semantic_dev_20260926/baseline_dev_20260928
cd "$ROOT" || exit 91
if [ -e "$BASE/four_methods_budget8_seed0_20260930.json" ]; then
  printf '%s\n' 'Sequence state exists; refusing duplicate start.'
  exit 92
fi
/project/guoqinglong/miniconda3/envs/cfbdm/bin/python -m scripts.run_four_detection_baselines \
  --sequence-id four_methods_budget8_seed0_20260930 \
  --budget-variant --batch-size 128 --time-cap-hours 8 \
  --code-commit 541e3389317f12e6f5f74d5c010341019c9ddb99 \
  > "$BASE/four_methods_budget8_seed0_20260930.console.log" 2>&1
result=$?
printf '%s\n' "$result" > "$BASE/four_methods_budget8_seed0_20260930.exitcode"
exit "$result"
