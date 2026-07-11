from __future__ import annotations

import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    width: int
    height: int
    annotations: Tuple[Tuple[int, float, float, float, float], ...]

    @property
    def classes(self) -> Tuple[int, ...]:
        return tuple(sorted({row[0] for row in self.annotations}))

    def has_class(self, class_id: int) -> bool:
        return class_id in self.classes


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_target_class(names: Sequence[str], target_name: str) -> int:
    matches = [index for index, name in enumerate(names) if name == target_name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one class named {target_name!r}, found {matches}")
    return matches[0]


def parse_voc(voc_root: Path, names: Sequence[str], split: str = "trainval") -> List[Sample]:
    class_to_id = {name: index for index, name in enumerate(names)}
    ids_path = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    image_ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples: List[Sample] = []
    for image_id in image_ids:
        root = ET.parse(voc_root / "Annotations" / f"{image_id}.xml").getroot()
        size = root.find("size")
        width = int(float(size.findtext("width")))
        height = int(float(size.findtext("height")))
        annotations = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name not in class_to_id:
                continue
            box = obj.find("bndbox")
            xmin = max(0.0, min(float(width), float(box.findtext("xmin"))))
            ymin = max(0.0, min(float(height), float(box.findtext("ymin"))))
            xmax = max(0.0, min(float(width), float(box.findtext("xmax"))))
            ymax = max(0.0, min(float(height), float(box.findtext("ymax"))))
            if xmax <= xmin or ymax <= ymin:
                continue
            annotations.append(
                (
                    class_to_id[name],
                    ((xmin + xmax) * 0.5) / width,
                    ((ymin + ymax) * 0.5) / height,
                    (xmax - xmin) / width,
                    (ymax - ymin) / height,
                )
            )
        if annotations:
            samples.append(
                Sample(
                    image_id=image_id,
                    image_path=voc_root / "JPEGImages" / f"{image_id}.jpg",
                    width=width,
                    height=height,
                    annotations=tuple(annotations),
                )
            )
    return samples


def _instance_counts(samples: Iterable[Sample], num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for sample in samples:
        for annotation in sample.annotations:
            counts[annotation[0]] += 1
    return counts


def _cooccurrence_counts(samples: Iterable[Sample], target_id: int, num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for sample in samples:
        if not sample.has_class(target_id):
            continue
        for class_id in sample.classes:
            if class_id != target_id:
                counts[class_id] += 1
    return counts


def _select_split(
    candidates: Sequence[Sample],
    size: int,
    target_id: int,
    person_fraction: float,
    seed: int,
    excluded: set[str],
) -> List[Sample]:
    rng = random.Random(seed)
    available = [sample for sample in candidates if sample.image_id not in excluded]
    person = [sample for sample in available if sample.has_class(target_id)]
    person_free = [sample for sample in available if not sample.has_class(target_id)]
    person.sort(key=lambda sample: (-len(sample.classes), sample.image_id))
    person_free.sort(key=lambda sample: (-len(sample.classes), sample.image_id))
    rng.shuffle(person)
    rng.shuffle(person_free)
    person_count = min(len(person), int(round(size * person_fraction)))
    selected = person[:person_count] + person_free[: size - person_count]
    if len(selected) < size:
        selected_ids = {sample.image_id for sample in selected}
        remainder = [sample for sample in available if sample.image_id not in selected_ids]
        rng.shuffle(remainder)
        selected.extend(remainder[: size - len(selected)])
    return sorted(selected, key=lambda sample: sample.image_id)


def build_split(
    catalog: Sequence[Sample], train_size: int, val_size: int, target_id: int, seed: int, num_classes: int
) -> Tuple[List[Sample], List[Sample], Dict]:
    val = _select_split(catalog, val_size, target_id, 0.5, seed + 1, set())
    used = {sample.image_id for sample in val}
    train = _select_split(catalog, train_size, target_id, 0.5, seed, used)

    def summary(samples: Sequence[Sample]) -> Dict:
        person_present = sum(sample.has_class(target_id) for sample in samples)
        cooccur = sum(sample.has_class(target_id) and len(sample.classes) > 1 for sample in samples)
        return {
            "file_count": len(samples),
            "person_present": person_present,
            "person_free": len(samples) - person_present,
            "person_cooccur": cooccur,
            "instance_count_by_class": _instance_counts(samples, num_classes),
            "person_cooccurrence_images_by_class": _cooccurrence_counts(samples, target_id, num_classes),
        }

    payload = {
        "seed": seed,
        "target_class_id": target_id,
        "train_ids": [sample.image_id for sample in train],
        "val_ids": [sample.image_id for sample in val],
        "train_summary": summary(train),
        "val_summary": summary(val),
        "selection": "deterministic 50% target-present / 50% target-free sampling; target-present pool includes cooccurrence images",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return train, val, payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_clean_dataset(
    train: Sequence[Sample], val: Sequence[Sample], root: Path, names: Sequence[str], target_id: int
) -> Dict:
    lists = {}
    for split_name, samples in (("train", train), ("val", val)):
        image_paths = []
        for sample in samples:
            image_path = root / "images" / split_name / sample.image_path.name
            label_path = root / "labels" / split_name / f"{sample.image_id}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if not image_path.exists() or file_sha256(image_path) != file_sha256(sample.image_path):
                shutil.copy2(sample.image_path, image_path)
            rows = [f"{cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}" for cls, x, y, w, h in sample.annotations]
            label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            image_paths.append(str(image_path.resolve()))
        list_path = root / f"{split_name}.txt"
        list_path.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
        lists[split_name] = list_path

    val_free = [sample for sample in val if not sample.has_class(target_id)]
    val_cooccur = [sample for sample in val if sample.has_class(target_id) and len(sample.classes) > 1]
    by_id = {sample.image_id: str((root / "images" / "val" / sample.image_path.name).resolve()) for sample in val}
    for name, samples in (("val_person_free", val_free), ("val_person_cooccur", val_cooccur)):
        path = root / f"{name}.txt"
        path.write_text("\n".join(by_id[sample.image_id] for sample in samples) + "\n", encoding="utf-8")
        lists[name] = path

    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": str(lists["train"].resolve()),
                "val": str(lists["val"].resolve()),
                "names": {index: name for index, name in enumerate(names)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {key: str(value.resolve()) for key, value in lists.items()} | {"dataset_yaml": str(yaml_path.resolve())}
