from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Iterable, Sequence

from ue_framework.data_utils import label_path_for_image, list_images, read_yolo_annotations


@dataclass(frozen=True)
class ImageRecord:
    source_id: str
    image_path: str
    annotations: tuple[dict, ...]

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(int(annotation["cls"]) for annotation in self.annotations)


@dataclass(frozen=True)
class Episode:
    support_clean: tuple[ImageRecord, ...]
    support_poison: tuple[ImageRecord, ...]
    query_clean: tuple[ImageRecord, ...]
    query_poison: tuple[ImageRecord, ...]
    support_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    target_class_ids: tuple[int, ...]
    non_target_class_ids: tuple[int, ...]
    class_counts: dict[int, dict[str, int]]
    class_validity: dict[int, bool]

    def validate(self, target_class_id: int) -> None:
        overlap = set(self.support_ids) & set(self.query_ids)
        if overlap:
            raise RuntimeError(f"support/query source ID overlap: {sorted(overlap)}")
        if self.support_ids != tuple(record.source_id for record in self.support_clean):
            raise RuntimeError("support IDs do not match clean records")
        if self.support_ids != tuple(record.source_id for record in self.support_poison):
            raise RuntimeError("support clean/poison pair identity mismatch")
        if self.query_ids != tuple(record.source_id for record in self.query_clean):
            raise RuntimeError("query IDs do not match clean records")
        if self.query_ids != tuple(record.source_id for record in self.query_poison):
            raise RuntimeError("query clean/poison pair identity mismatch")
        for name, records in (("support", self.support_clean), ("query", self.query_clean)):
            if not any(int(target_class_id) in record.class_ids for record in records):
                raise RuntimeError(f"target class absent from {name}")


def load_records(dataset_root: str, images: str, labels: str) -> list[ImageRecord]:
    image_dir = os.path.join(dataset_root, images)
    label_dir = os.path.join(dataset_root, labels)
    records: list[ImageRecord] = []
    for image_path in list_images(image_dir):
        source_id = os.path.splitext(os.path.basename(image_path))[0]
        annotations = tuple(read_yolo_annotations(label_path_for_image(image_path, label_dir)))
        records.append(ImageRecord(source_id, image_path, annotations))
    if len({record.source_id for record in records}) != len(records):
        raise RuntimeError("duplicate source IDs in dataset")
    return records


def _class_counts(records: Sequence[ImageRecord], class_id: int) -> int:
    return sum(sum(int(annotation["cls"]) == int(class_id) for annotation in record.annotations) for record in records)


class DisjointEpisodeSampler:
    def __init__(
        self,
        records: Iterable[ImageRecord],
        target_class_id: int,
        num_classes: int = 20,
        support_size: int = 2,
        query_size: int = 2,
        minimum_class_samples: int = 1,
        seed: int = 0,
    ) -> None:
        self.records = tuple(records)
        self.target_class_id = int(target_class_id)
        self.num_classes = int(num_classes)
        self.support_size = int(support_size)
        self.query_size = int(query_size)
        self.minimum_class_samples = int(minimum_class_samples)
        self.seed = int(seed)
        if self.support_size <= 0 or self.query_size <= 0:
            raise ValueError("support_size and query_size must be positive")
        if self.minimum_class_samples <= 0:
            raise ValueError("minimum_class_samples must be positive")

    def sample(self, episode_index: int = 0, worker_id: int = 0) -> Episode:
        candidates = [record for record in self.records if self.target_class_id in record.class_ids]
        required = self.support_size + self.query_size
        if len(candidates) < required:
            raise RuntimeError(
                f"insufficient distinct target images: required={required}, available={len(candidates)}; reuse is forbidden"
            )
        generator = random.Random(self.seed + 1000003 * int(episode_index) + 9176 * int(worker_id))
        order = list(range(len(candidates)))
        generator.shuffle(order)
        selected = [candidates[index] for index in order[:required]]
        support = tuple(selected[: self.support_size])
        query = tuple(selected[self.support_size :])
        class_counts: dict[int, dict[str, int]] = {}
        class_validity: dict[int, bool] = {}
        non_target_ids: list[int] = []
        for class_id in range(self.num_classes):
            support_count = _class_counts(support, class_id)
            query_count = _class_counts(query, class_id)
            class_counts[class_id] = {"support": support_count, "query": query_count}
            valid = (
                class_id != self.target_class_id
                and support_count >= self.minimum_class_samples
                and query_count >= self.minimum_class_samples
            )
            class_validity[class_id] = valid
            if valid:
                non_target_ids.append(class_id)
        episode = Episode(
            support_clean=support,
            support_poison=support,
            query_clean=query,
            query_poison=query,
            support_ids=tuple(record.source_id for record in support),
            query_ids=tuple(record.source_id for record in query),
            target_class_ids=(self.target_class_id,),
            non_target_class_ids=tuple(non_target_ids),
            class_counts=class_counts,
            class_validity=class_validity,
        )
        episode.validate(self.target_class_id)
        return episode


def preserve_source_id(record: ImageRecord, _augmentation_name: str) -> str:
    """Augmentations operate on pixels; source identity remains immutable."""
    return record.source_id

