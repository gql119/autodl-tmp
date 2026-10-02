"""Confirm that B2-TO-Safe differs from formal B2-TO only as preregistered."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline_path, candidate_path = Path(args.baseline), Path(args.candidate)
    baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    normalized = json.loads(json.dumps(candidate))
    normalized["platform"]["run_root"] = baseline["platform"]["run_root"]
    current = normalized["methods"]["legacy_kproto_ret"]["semantic"]
    original = baseline["methods"]["legacy_kproto_ret"]["semantic"]
    current["artifact_dir"] = original["artifact_dir"]
    current.pop("strict_tsfl", None)
    current.pop("safe_bg", None)
    differences = []
    def walk(left, right, prefix=""):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                path = f"{prefix}.{key}" if prefix else key
                if key not in left or key not in right:
                    differences.append(path)
                else:
                    walk(left[key], right[key], path)
        elif left != right:
            differences.append(prefix)
    walk(baseline, normalized)
    if differences:
        raise RuntimeError("Unapproved B2-TO config differences: " + ", ".join(differences))
    result = {
        "state": "passed", "baseline_config_sha256": sha256(baseline_path),
        "candidate_config_sha256": sha256(candidate_path),
        "only_training_config_changes": [
            "platform.run_root", "semantic.artifact_dir", "semantic.strict_tsfl", "semantic.safe_bg"
        ],
        "lambda_orth": current["lambda_orth"], "orth_worst_k": current["orth_worst_k"],
        "noise_epochs": candidate["methods"]["legacy_kproto_ret"]["universal_epochs"],
        "noise_steps_expected": math.ceil(6095 / candidate["methods"]["legacy_kproto_ret"]["universal_batch_size"])
                                * candidate["methods"]["legacy_kproto_ret"]["universal_epochs"],
        "victim_epochs": candidate["victim"]["epochs"], "victim_batch": candidate["victim"]["batch"],
        "seed": candidate["experiment"]["seeds"][0],
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
