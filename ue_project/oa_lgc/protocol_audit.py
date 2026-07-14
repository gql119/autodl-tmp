from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import yaml


VOC_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _class_distribution(annotation_root: Path, source_ids: list[str]) -> list[dict]:
    image_counts = {name: 0 for name in VOC_NAMES}
    instance_counts = {name: 0 for name in VOC_NAMES}
    for source_id in source_ids:
        tree = ET.parse(annotation_root / f"{source_id}.xml")
        present = set()
        for obj in tree.findall("object"):
            name = obj.findtext("name")
            if name in instance_counts:
                instance_counts[name] += 1
                present.add(name)
        for name in present:
            image_counts[name] += 1
    return [
        {
            "class_id": class_id,
            "class_name": name,
            "image_count": image_counts[name],
            "instance_count": instance_counts[name],
        }
        for class_id, name in enumerate(VOC_NAMES)
    ]


def run(config_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    manifests = output / "manifests"
    manifests.mkdir(exist_ok=False)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {name: Path(value) for name, value in config["paths"].items()}
    voc2007 = paths["voc2007_root"]
    voc2012 = paths["voc2012_root"]
    voc2007_trainval = voc2007 / "ImageSets" / "Main" / "trainval.txt"
    voc2007_test = voc2007 / "ImageSets" / "Main" / "test.txt"
    voc2012_trainval = voc2012 / "ImageSets" / "Main" / "trainval.txt"
    available_ids = _ids(voc2007_trainval)
    mini_payload = json.loads(paths["mini_split_manifest"].read_text(encoding="utf-8"))
    mini_train_ids = [str(value) for value in mini_payload["train_ids"]]
    mini_val_ids = [str(value) for value in mini_payload["val_ids"]]
    (manifests / "available_voc2007_trainval_manifest.txt").write_text(
        "\n".join(available_ids) + "\n", encoding="utf-8"
    )
    (manifests / "historical_mini_train_manifest.txt").write_text(
        "\n".join(mini_train_ids) + "\n", encoding="utf-8"
    )
    (manifests / "historical_mini_val_manifest.txt").write_text(
        "\n".join(mini_val_ids) + "\n", encoding="utf-8"
    )
    missing = {
        "status": "blocked",
        "exact_train_manifest_generated": False,
        "exact_test_manifest_generated": False,
        "missing_required_components": [
            name
            for name, exists in (
                ("VOC2012 trainval", voc2012_trainval.is_file()),
                ("VOC2007 test", voc2007_test.is_file()),
            )
            if not exists
        ],
        "reason": "The required protocol cannot be reconstructed without guessing or downloading data.",
    }
    (manifests / "missing_required_components.json").write_text(
        json.dumps(missing, indent=2) + "\n", encoding="utf-8"
    )
    overlap = {
        "exact_protocol_train_test_overlap": None,
        "exact_protocol_overlap_auditable": False,
        "historical_mini_train_val_overlap_count": len(set(mini_train_ids) & set(mini_val_ids)),
        "historical_mini_train_count": len(mini_train_ids),
        "historical_mini_val_count": len(mini_val_ids),
        "available_voc2007_trainval_count": len(available_ids),
    }
    (manifests / "overlap_audit.json").write_text(
        json.dumps(overlap, indent=2) + "\n", encoding="utf-8"
    )
    available_hashes = {
        "voc2007_trainval_split_sha256": _sha256(voc2007_trainval),
        "mini_split_manifest_sha256": _sha256(paths["mini_split_manifest"]),
        "note": "These hashes cover available components only; they are not hashes of the required C4 protocol.",
    }
    (manifests / "available_component_hashes.json").write_text(
        json.dumps(available_hashes, indent=2) + "\n", encoding="utf-8"
    )
    blocked_hashes = {
        "status": "blocked",
        "manifest_generated": False,
        "hashes": None,
        "reason": missing["reason"],
    }
    (manifests / "train_hashes.json").write_text(
        json.dumps({**blocked_hashes, "missing": "VOC2012 trainval"}, indent=2) + "\n", encoding="utf-8"
    )
    (manifests / "test_hashes.json").write_text(
        json.dumps({**blocked_hashes, "missing": "VOC2007 test"}, indent=2) + "\n", encoding="utf-8"
    )
    distribution = _class_distribution(voc2007 / "Annotations", available_ids)
    with (manifests / "available_voc2007_trainval_class_distribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(distribution[0]))
        writer.writeheader()
        writer.writerows(distribution)

    clean_config = {
        "status": "blocked",
        "required_protocol": config["required_protocol"],
        "historical_mini_config": yaml.safe_load(paths["historical_clean_config"].read_text(encoding="utf-8")),
        "historical_baseline_protocol_eligible": False,
    }
    (output / "clean_config.yaml").write_text(
        yaml.safe_dump(clean_config, sort_keys=False), encoding="utf-8"
    )
    with paths["historical_training_metrics"].open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    with (output / "clean_training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["protocol_eligible", "source_protocol", *rows[0].keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "protocol_eligible": 0,
                    "source_protocol": "historical_VOC2007_trainval_mini_800_200",
                    **row,
                }
            )
    historical_eval = json.loads(paths["historical_eval_metrics"].read_text(encoding="utf-8"))
    clean_eval = {
        "status": "historical_evidence_only_not_C4_baseline",
        "protocol_eligible": False,
        "ineligibility_reasons": [
            "training is an 800-image subset of VOC2007 trainval",
            "evaluation is a 200-image subset of VOC2007 trainval, not VOC2007 test",
            "VOC2012 trainval is absent",
            "per-epoch target and non-target AP curves are absent",
        ],
        "historical_metrics": historical_eval,
        "E_pilot": None,
    }
    (output / "clean_eval_metrics.json").write_text(
        json.dumps(clean_eval, indent=2) + "\n", encoding="utf-8"
    )
    with paths["historical_classwise_ap"].open("r", newline="", encoding="utf-8") as source:
        class_rows = list(csv.DictReader(source))
    with (output / "classwise_ap.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["protocol_eligible", "source_protocol", *class_rows[0].keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in class_rows:
            writer.writerow(
                {
                    "protocol_eligible": 0,
                    "source_protocol": "historical_VOC2007_trainval_mini_800_200",
                    **row,
                }
            )
    checkpoint = paths["historical_best_checkpoint"]
    checkpoints = {
        "historical_mini_best": {
            "path": str(checkpoint),
            "exists": checkpoint.is_file(),
            "sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
            "recorded_sha256": historical_eval.get("checkpoint_sha256"),
            "hash_matches_record": checkpoint.is_file()
            and _sha256(checkpoint) == historical_eval.get("checkpoint_sha256"),
            "protocol_eligible": False,
            "reason": "Checkpoint belongs to the historical mini split, not the required C4 protocol.",
        }
    }
    (output / "checkpoints_manifest.json").write_text(
        json.dumps(checkpoints, indent=2) + "\n", encoding="utf-8"
    )
    gate = {
        "protocol_auditable": False,
        "train_test_overlap_zero": None,
        "clean_baseline_reproducible": False,
        "clean_target_non_target_converged": False,
        "E_pilot_selected_from_curve": False,
        "source_clear_checkpoint_is_not_sole_baseline": True,
    }
    summary = {
        "status": "blocked",
        "gate": gate,
        "missing_required_components": missing["missing_required_components"],
        "available_voc2007_trainval": len(available_ids),
        "historical_mini_train": len(mini_train_ids),
        "historical_mini_val": len(mini_val_ids),
        "historical_checkpoint_hash_verified": checkpoints["historical_mini_best"]["hash_matches_record"],
        "C5_allowed": False,
        "silent_download_performed": False,
        "test_data_used_for_optimization": False,
    }
    (output / "run.log").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the OA-LGC C4 VOC protocol without downloading data.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(run(arguments.config, arguments.output), indent=2))
