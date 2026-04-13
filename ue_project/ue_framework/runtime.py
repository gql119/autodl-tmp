import os
from dataclasses import dataclass
from typing import Dict

from .paths import RunPaths


@dataclass
class RunContext:
    cfg: Dict
    method: str
    steps: int
    seed: int
    stage: str
    gpu_id: int
    platform_mode: str
    paths: RunPaths

    @property
    def dataset_root(self) -> str:
        return self.cfg["data"]["dataset_root"]

    @property
    def train_img_dir(self) -> str:
        return os.path.join(self.dataset_root, self.cfg["data"]["train_images"])

    @property
    def train_label_dir(self) -> str:
        return os.path.join(self.dataset_root, self.cfg["data"]["train_labels"])

    @property
    def val_img_dir(self) -> str:
        return os.path.join(self.dataset_root, self.cfg["data"]["val_images"])

    @property
    def val_label_dir(self) -> str:
        return os.path.join(self.dataset_root, self.cfg["data"]["val_labels"])

