"""Runtime verification of the dataset identity bound by protocol_lock.json."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ue_framework.data_utils import list_images


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_fingerprint(paths: list[Path]) -> str:
    lines = [f"{sha256_file(path)}  {path.resolve()}\n" for path in sorted(paths)]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def locked_source_inventory(lock: dict, data_root: str | Path) -> tuple[dict, dict]:
    source = Path(data_root).resolve()
    if source != Path(lock["dataset"]["root"]).resolve():
        raise RuntimeError("Dataset root differs from the protocol lock")
    result: dict[str, dict] = {}
    for split in ("train", "val"):
        images = [Path(p) for p in list_images(str(source / "images" / split))]
        labels = sorted((source / "labels" / split).glob("*.txt"))
        expected_images = int(lock["dataset"][f"{split}_images"])
        expected_labels = int(lock["dataset"][f"{split}_labels"])
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if len(images) != expected_images or len(labels) != expected_labels:
            raise RuntimeError(f"Unexpected locked {split} counts: {len(images)} images, {len(labels)} labels")
        if len(image_stems) != expected_images or image_stems != label_stems:
            raise RuntimeError(f"Locked {split} image/label stems are not one-to-one")
        fingerprint = split_fingerprint(images + labels)
        expected = lock["dataset"][f"{split}_split_fingerprint"]
        if fingerprint != expected:
            raise RuntimeError(f"{split} source fingerprint differs from lock: {fingerprint}")
        result[split] = {"images": images, "labels": labels, "fingerprint": fingerprint}
    combined = hashlib.sha256((result["train"]["fingerprint"] + "\n" +
                               result["val"]["fingerprint"]).encode("utf-8")).hexdigest()
    if combined != lock["dataset"]["fingerprint"]:
        raise RuntimeError("Combined dataset fingerprint differs from protocol lock")

    target_images = target_objects = 0
    for label in result["train"]["labels"]:
        count = 0
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields and int(float(fields[0])) == int(lock["target"]["class_id"]):
                count += 1
        target_images += int(count > 0)
        target_objects += count
    if (target_images != int(lock["dataset"]["target_images"]) or
            target_objects != int(lock["dataset"]["target_objects"])):
        raise RuntimeError("Locked target image/object counts differ from the source labels")
    inventory = {
        split: {"images": result[split]["images"], "labels": result[split]["labels"],
                "fingerprint": result[split]["fingerprint"]}
        for split in result
    }
    summary = {
        "train_images": len(result["train"]["images"]),
        "train_labels": len(result["train"]["labels"]),
        "val_images": len(result["val"]["images"]),
        "val_labels": len(result["val"]["labels"]),
        "target_images": target_images,
        "target_objects": target_objects,
        "train_fingerprint": result["train"]["fingerprint"],
        "val_fingerprint": result["val"]["fingerprint"],
        "dataset_fingerprint": combined,
    }
    return inventory, summary
