"""Compare the clean victim's production args with a completed B2-T victim."""

import json
from pathlib import Path

from run_locked_clean_baseline import ROOT, check_installed_protocol, locked_train_args, sha256_file


REFERENCE = Path(
    "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
    "b2_adaptive_dev_20260926/semantic_dev_20260926/"
    "victim_b2_t_seed0_20260926/run_spec.json"
)


def main() -> None:
    lock = json.loads((ROOT / "protocol_lock.json").read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    expected = reference["train_args"]
    actual = locked_train_args(lock, ROOT / "clean_protocol_check_unused",
                               ROOT / "clean_protocol_check_unused.yaml", smoke=False)
    skip_paths = {"data", "project"}
    differences = {key: {"reference": expected.get(key), "clean": actual.get(key)}
                   for key in sorted((set(expected) | set(actual)) - skip_paths)
                   if expected.get(key) != actual.get(key)}
    if differences:
        raise RuntimeError(f"Clean victim training arguments differ: {differences}")
    init = Path(lock["victim"]["init_yaml"])
    if sha256_file(init) != reference["init_yaml_sha256"]:
        raise RuntimeError("Architecture YAML differs from reference victim")
    version = check_installed_protocol(lock)
    print(json.dumps({"state": "passed", "reference_run_spec": str(REFERENCE),
                      "matching_argument_count": len(set(expected) - skip_paths),
                      "ignored_output_paths": sorted(skip_paths),
                      "init_yaml_sha256": sha256_file(init),
                      "ultralytics_version": version}, indent=2))


if __name__ == "__main__":
    main()
