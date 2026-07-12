import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def image_ids(directory):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted({os.path.splitext(name)[0] for name in os.listdir(directory) if os.path.splitext(name)[1].lower() in extensions})


def overlap_rows(sets):
    rows = []
    names = list(sets)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            overlap = sorted(set(sets[first]) & set(sets[second]))
            rows.extend({"first": first, "second": second, "image_id": item} for item in overlap)
    return rows


def audit_sets(checkpoint_train, checkpoint_val, mini_train, mini_val):
    known = checkpoint_train is not None and checkpoint_val is not None
    summary = {
        "checkpoint_train_metadata_available": checkpoint_train is not None,
        "checkpoint_val_metadata_available": checkpoint_val is not None,
        "mini_train_count": len(mini_train),
        "mini_val_count": len(mini_val),
        "mini_train_mini_val_overlap": len(set(mini_train) & set(mini_val)),
        "checkpoint_train_mini_val_overlap": None if checkpoint_train is None else len(set(checkpoint_train) & set(mini_val)),
        "checkpoint_val_mini_val_overlap": None if checkpoint_val is None else len(set(checkpoint_val) & set(mini_val)),
    }
    if summary["mini_train_mini_val_overlap"]:
        conclusion = "evaluation leakage risk"
    elif not known:
        conclusion = "metadata insufficient"
    elif summary["checkpoint_train_mini_val_overlap"] or summary["checkpoint_val_mini_val_overlap"]:
        conclusion = "partial overlap"
    else:
        conclusion = "clean checkpoint independent"
    summary["conclusion"] = conclusion
    summary["evaluation_leakage_risk"] = conclusion != "clean checkpoint independent"
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mini-root", required=True)
    args = parser.parse_args()
    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("train_args", {})
    metadata = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "sha256": hashlib.sha256(open(args.checkpoint, "rb").read()).hexdigest(),
        "epoch": checkpoint.get("epoch"),
        "train_args": train_args,
        "names": getattr(checkpoint.get("model"), "names", None),
        "embedded_train_ids": None,
        "embedded_val_ids": None,
        "data_yaml": train_args.get("data"),
        "data_yaml_available_locally": bool(train_args.get("data") and os.path.isfile(str(train_args.get("data")))),
        "stage0_role": "frozen surrogate used for feature/gradient statistics and fixed-Q construction",
        "stage1_role": "frozen surrogate used for poison optimization; not the scratch victim initialization",
        "resume_baseline_role": "evaluation-only checkpoint; independence from mini val is not established",
    }
    mini_train = image_ids(os.path.join(args.mini_root, "images", "train"))
    mini_val = image_ids(os.path.join(args.mini_root, "images", "val"))
    checkpoint_train = metadata["embedded_train_ids"]
    checkpoint_val = metadata["embedded_val_ids"]
    summary = audit_sets(checkpoint_train, checkpoint_val, mini_train, mini_val)
    for name, values in [("train_ids.txt", checkpoint_train or []), ("val_ids.txt", checkpoint_val or []), ("mini_train_ids.txt", mini_train), ("mini_val_ids.txt", mini_val)]:
        with open(os.path.join(output, name), "w", encoding="utf-8") as file:
            file.write("\n".join(values) + ("\n" if values else ""))
    sets = {"mini_train": mini_train, "mini_val": mini_val}
    if checkpoint_train is not None:
        sets["checkpoint_train"] = checkpoint_train
    if checkpoint_val is not None:
        sets["checkpoint_val"] = checkpoint_val
    rows = overlap_rows(sets)
    with open(os.path.join(output, "overlap_details.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["first", "second", "image_id"]); writer.writeheader(); writer.writerows(rows)
    for name, value in [("checkpoint_metadata.json", metadata), ("overlap_summary.json", summary)]:
        with open(os.path.join(output, name), "w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, ensure_ascii=False, default=str)
    open(os.path.join(output, "command.txt"), "w", encoding="utf-8").write(" ".join(sys.argv) + "\n")
    open(os.path.join(output, "environment.txt"), "w", encoding="utf-8").write(f"platform={platform.platform()}\npython={platform.python_version()}\ntorch={torch.__version__}\n")
    open(os.path.join(output, "git_commit.txt"), "w", encoding="utf-8").write(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
