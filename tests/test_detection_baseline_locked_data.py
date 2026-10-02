"""Runtime lock fingerprints and full materialization integrity contract."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ue_framework.baselines_det.locked_data import locked_source_inventory, sha256_file, split_fingerprint
from scripts.materialize_detection_baseline import materialize
from scripts.generate_detection_baseline import expected_cycle_counters


class LockedDetectionDataTests(unittest.TestCase):
    def test_frozen_batch128_full_cycle_counter_contract(self):
        self.assertEqual(expected_cycle_counters("em_det", 48), {
            "forward_calls": 7065, "backward_calls": 970, "optimizer_steps": 10,
            "pgd_steps": 0, "noise_steps": 121900, "eot_samples": 0,
            "stability_samples": 0, "push_steps": 0,
        })
        self.assertEqual(expected_cycle_counters("rem_det", 48), {
            "forward_calls": 26928, "backward_calls": 26928, "optimizer_steps": 48,
            "pgd_steps": 3108450, "noise_steps": 0, "eot_samples": 304750,
            "stability_samples": 0, "push_steps": 0,
        })
        self.assertEqual(expected_cycle_counters("entf_push_det", 48), {
            "forward_calls": 14448, "backward_calls": 14400, "optimizer_steps": 0,
            "pgd_steps": 0, "noise_steps": 0, "eot_samples": 0,
            "stability_samples": 0, "push_steps": 1828500,
        })
        self.assertEqual(expected_cycle_counters("sem_det", 48), {
            "forward_calls": 2928, "backward_calls": 2928, "optimizer_steps": 48,
            "pgd_steps": 60950, "noise_steps": 0, "eot_samples": 0,
            "stability_samples": 304750, "push_steps": 0,
        })

    def _fixture(self, root: Path):
        source = root / "voc"
        for split in ("train", "val"):
            (source / "images" / split).mkdir(parents=True)
            (source / "labels" / split).mkdir(parents=True)
        target = np.full((16, 16, 3), 60, dtype=np.uint8)
        Image.fromarray(target).save(source / "images/train/target.png")
        Image.fromarray(np.full((16, 16, 3), 80, dtype=np.uint8)).save(source / "images/train/clean.png")
        Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8)).save(source / "images/val/val.png")
        (source / "labels/train/target.txt").write_text(
            "14 0.5 0.5 0.5 0.5\n1 0.625 0.5 0.25 0.25\n", encoding="utf-8")
        (source / "labels/train/clean.txt").write_text("1 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        (source / "labels/val/val.txt").write_text("14 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        fingerprints = {}
        for split in ("train", "val"):
            files = list((source / "images" / split).glob("*")) + list((source / "labels" / split).glob("*.txt"))
            fingerprints[f"{split}_split_fingerprint"] = split_fingerprint(files)
        combined = hashlib.sha256((fingerprints["train_split_fingerprint"] + "\n" +
                                   fingerprints["val_split_fingerprint"]).encode()).hexdigest()
        lock = {
            "dataset": {
                "root": str(source.resolve()), "fingerprint": combined,
                **fingerprints, "train_images": 2, "train_labels": 2,
                "val_images": 1, "val_labels": 1, "target_images": 1,
                "target_objects": 1,
            },
            "target": {"class_id": 14},
        }
        return source, lock

    def test_source_inventory_recomputes_file_fingerprints(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, lock = self._fixture(Path(temporary))
            inventory, summary = locked_source_inventory(lock, source)
            self.assertEqual(summary["target_images"], 1)
            self.assertEqual(summary["target_objects"], 1)
            self.assertEqual(len(inventory["train"]["images"]), 2)
            (source / "images/train/clean.png").write_bytes(b"mutated")
            with self.assertRaisesRegex(RuntimeError, "fingerprint differs"):
                locked_source_inventory(lock, source)

    def test_materializer_preserves_clean_data_and_applies_only_person_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, lock = self._fixture(root)
            inventory, _ = locked_source_inventory(lock, source)
            protocol = root / "protocol_lock.json"
            protocol.write_text(json.dumps(lock), encoding="utf-8")
            generation = root / "generation"
            noise = generation / "noise"
            noise.mkdir(parents=True)
            clean = np.asarray(Image.open(source / "images/train/target.png").convert("RGB"), dtype=np.uint8)
            poisoned = clean.copy()
            poisoned[5, 5] += 8
            poisoned[6, 8] += 4
            Image.fromarray(poisoned).save(noise / "target.png")
            (generation / "status.json").write_text(json.dumps({
                "state": "complete", "completed_target_images": 1,
                "dataset_fingerprint": lock["dataset"]["fingerprint"],
                "protocol_lock_sha256": sha256_file(protocol),
                "method": "em_det", "run_id": "tiny_test_run",
            }), encoding="utf-8")
            (generation / "run_spec.json").write_text(json.dumps({
                "state": "complete", "method": "em_det", "run_id": "tiny_test_run",
                "source_file_sha256": {"test_fixture": "not-production-evidence"},
            }), encoding="utf-8")
            output = root / "materialization"
            output.mkdir()
            result = materialize(lock, inventory, generation, output, "em_det", "tiny_test_run", protocol)
            data = output / "poisoned_data"
            self.assertEqual(result["state"], "passed")
            self.assertEqual(result["changed_person_gt_boxes"], 1)
            self.assertEqual(result["max_abs_delta_byte"], {"person_only": 8, "person_overlap": 4})
            self.assertTrue((data / "images/train/clean.png").read_bytes() ==
                            (source / "images/train/clean.png").read_bytes())
            self.assertTrue((data / "images/val/val.png").read_bytes() ==
                            (source / "images/val/val.png").read_bytes())
            self.assertEqual((data / "labels/train/target.txt").read_bytes(),
                             (source / "labels/train/target.txt").read_bytes())


if __name__ == "__main__":
    unittest.main()
