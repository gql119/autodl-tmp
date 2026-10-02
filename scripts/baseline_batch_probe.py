"""Isolated production-path smoke and exact-inner-unit throughput probe."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from ue_framework.baselines_det.core import EPS, file_sha256, model_state_sha256, poison_image
from ue_framework.baselines_det.production import (
    BatchItem, BatchedPersonROIFeatures, em_noise_batch, em_train_batch,
    entf_centroid_batch, entf_push_batch, make_item, new_counters,
    rem_defender_batch, sem_defender_batch,
)
from ue_framework.data_utils import list_images, label_path_for_image, read_yolo_annotations
from ue_framework.baselines_det.core import TARGET_ID
from scripts.generate_detection_baseline import (
    ALLOWED_ROOT, DEFENDER_STEPS, DEFENDER_STEP, EM_NOISE_STEP, EM_NOISE_STEPS,
    EM_TRAIN_BATCHES, EOT_SAMPLES, ENTF_STEP, ENTF_STEPS, INNER_RADIUS,
    INNER_STEP, INNER_STEPS, IMG_SIZE, PERSON_COUNT, _assert_lock, load_batch, source_hashes,
)


METHODS = ("em_det", "rem_det", "entf_push_det", "sem_det")


def item_index(data_root: Path, lock_path: Path, surrogate: Path, count: int):
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    images, labels, targets = _assert_lock(lock, surrogate, data_root)
    random.Random(0).shuffle(targets)
    return lock, labels, targets[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "two-block-smoke", "exact-inner"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--surrogate", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--full-centroid", action="store_true")
    parser.add_argument("--code-commit", default="unspecified")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.relative_to(ALLOWED_ROOT)
    except ValueError as exc:
        raise PermissionError("Probe outputs must remain in the authorized baseline root") from exc
    if output.exists() or args.batch_size < 1:
        raise FileExistsError(f"Probe output collision or invalid batch size: {output}")
    if args.full_centroid and args.method != "entf_push_det":
        raise ValueError("--full-centroid is only relevant for EntF")
    start = time.time()
    lock, labels, target_paths = item_index(args.dataset_root, args.protocol_lock,
                                             args.surrogate,
                                             PERSON_COUNT if args.full_centroid else args.batch_size)
    if len(target_paths) < args.batch_size:
        raise ValueError("Insufficient target samples for requested probe batch")
    target_paths = target_paths[:args.batch_size]
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    np.random.seed(0)
    model = YOLO(str(args.surrogate)).model.to(device)
    if int(model.nc) != 20:
        raise RuntimeError("Probe detector must be VOC20")
    if args.method == "entf_push_det":
        extractor = BatchedPersonROIFeatures(model, imgsz=IMG_SIZE)
        optimizer = None
    else:
        extractor = None
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    items = load_batch(target_paths, labels, device)
    counters = new_counters()
    loss_observations = []
    before = model_state_sha256(model)
    torch.cuda.reset_peak_memory_stats(device)
    begin_update = time.time()
    if args.mode in ("smoke", "two-block-smoke"):
        if args.method == "em_det":
            loss_observations.append(em_train_batch(model, optimizer, items, imgsz=IMG_SIZE, counters=counters))
            deltas = em_noise_batch(model, items, steps=2, step_size=EM_NOISE_STEP,
                                    imgsz=IMG_SIZE, counters=counters,
                                    loss_trace=loss_observations)
        elif args.method == "rem_det":
            deltas, _ = rem_defender_batch(
                model, optimizer, items, defender_steps=1, eot_samples=1, pgd_steps=1,
                defender_step_size=DEFENDER_STEP, adversarial_radius=INNER_RADIUS,
                adversarial_step_size=INNER_STEP, imgsz=IMG_SIZE, seed=0,
                counters=counters, loss_trace=loss_observations)
        elif args.method == "sem_det":
            deltas, _ = sem_defender_batch(
                model, optimizer, items, defender_steps=1, stability_samples=1,
                radius=INNER_RADIUS, defender_step_size=DEFENDER_STEP,
                surrogate_pgd_steps=1, imgsz=IMG_SIZE, seed=0, counters=counters,
                loss_trace=loss_observations)
        else:
            # Reach the production centroid and Push sinks on training-only
            # images; 2 Push steps are smoke-only and are not a formal schedule.
            centroid, object_count, centroid_batches = entf_centroid_batch(extractor, [items])
            counters["forward_calls"] += centroid_batches
            deltas = entf_push_batch(extractor, items, centroid, steps=2,
                                     step_size=ENTF_STEP, seed=0, counters=counters,
                                     loss_trace=loss_observations)
    else:
        if args.method == "em_det":
            for _ in range(EM_TRAIN_BATCHES):
                loss_observations.append(em_train_batch(model, optimizer, items, imgsz=IMG_SIZE, counters=counters))
            deltas = em_noise_batch(model, items, steps=EM_NOISE_STEPS,
                                    step_size=EM_NOISE_STEP, imgsz=IMG_SIZE, counters=counters,
                                    loss_trace=loss_observations)
        elif args.method == "rem_det":
            deltas, _ = rem_defender_batch(
                model, optimizer, items, defender_steps=DEFENDER_STEPS,
                eot_samples=EOT_SAMPLES, pgd_steps=INNER_STEPS,
                defender_step_size=DEFENDER_STEP, adversarial_radius=INNER_RADIUS,
                adversarial_step_size=INNER_STEP, imgsz=IMG_SIZE, seed=0,
                counters=counters, loss_trace=loss_observations)
        elif args.method == "sem_det":
            deltas, _ = sem_defender_batch(
                model, optimizer, items, defender_steps=DEFENDER_STEPS,
                stability_samples=EOT_SAMPLES, radius=INNER_RADIUS,
                defender_step_size=DEFENDER_STEP,
                surrogate_pgd_steps=INNER_STEPS, imgsz=IMG_SIZE, seed=0,
                counters=counters, loss_trace=loss_observations)
        else:
            if args.full_centroid:
                image_paths = list_images(str(Path(args.dataset_root) / "images" / "train"))
                full_target = []
                object_count = 0
                for image_path in image_paths:
                    annotations = read_yolo_annotations(label_path_for_image(image_path, str(labels)))
                    if any(int(a["cls"]) == TARGET_ID for a in annotations):
                        full_target.append(Path(image_path))
                        object_count += sum(int(a["cls"]) == TARGET_ID for a in annotations)
                if len(full_target) != PERSON_COUNT:
                    raise RuntimeError("Full centroid index does not contain all target images")
                def centroid_batches():
                    for offset in range(0, len(full_target), args.batch_size):
                        batch = load_batch(full_target[offset:offset + args.batch_size], labels, device)
                        yield batch
                centroid, seen_objects, n_batches = entf_centroid_batch(extractor, centroid_batches())
                if seen_objects != object_count or object_count != lock["dataset"]["target_objects"]:
                    raise RuntimeError("Full EntF centroid object count differs from protocol lock")
                counters["forward_calls"] += n_batches
                centroid_identity = {"person_objects": object_count,
                                     "centroid_forward_batches": n_batches}
            else:
                centroid, object_count, n_batches = entf_centroid_batch(extractor, [items])
                counters["forward_calls"] += n_batches
                centroid_identity = {"person_objects": object_count,
                                     "centroid_forward_batches": n_batches,
                                     "full_centroid": False}
            deltas = entf_push_batch(extractor, items, centroid, steps=ENTF_STEPS,
                                     step_size=ENTF_STEP, seed=0, counters=counters,
                                     loss_trace=loss_observations)
    first_block_deltas = [delta.detach().clone() for delta in deltas]
    if args.mode == "two-block-smoke":
        continued_items = [BatchItem(x.clean, x.annotations, x.support, d, x.stem)
                           for x, d in zip(items, first_block_deltas)]
        if args.method == "em_det":
            loss_observations.append(em_train_batch(model, optimizer, continued_items,
                                                    imgsz=IMG_SIZE, counters=counters))
            deltas = em_noise_batch(model, continued_items, steps=2,
                                    step_size=EM_NOISE_STEP, imgsz=IMG_SIZE,
                                    counters=counters, loss_trace=loss_observations)
        elif args.method == "rem_det":
            deltas, _ = rem_defender_batch(
                model, optimizer, continued_items, defender_steps=1, eot_samples=1,
                pgd_steps=1, defender_step_size=DEFENDER_STEP,
                adversarial_radius=INNER_RADIUS, adversarial_step_size=INNER_STEP,
                imgsz=IMG_SIZE, seed=1, counters=counters,
                loss_trace=loss_observations)
        elif args.method == "sem_det":
            deltas, _ = sem_defender_batch(
                model, optimizer, continued_items, defender_steps=1,
                stability_samples=1, radius=INNER_RADIUS,
                defender_step_size=DEFENDER_STEP, surrogate_pgd_steps=1,
                imgsz=IMG_SIZE, seed=1, counters=counters,
                loss_trace=loss_observations)
        else:
            deltas = entf_push_batch(
                extractor, continued_items, centroid, steps=2, step_size=ENTF_STEP,
                seed=1, counters=counters, loss_trace=loss_observations,
                initial_deltas=first_block_deltas)
    elapsed = time.time() - begin_update
    finite_delta = all(torch.isfinite(delta).all().item() for delta in deltas)
    finite_loss = bool(loss_observations) and bool(np.isfinite(np.asarray(loss_observations)).all())
    displacement = float(torch.stack([x.abs().mean() for x in deltas]).mean().detach())
    state_after = model_state_sha256(model)
    item_reports = []
    for item, delta in zip(items, deltas):
        poison = poison_image(item.clean, delta, item.support)
        delta_effect = (poison - item.clean).abs()
        item_reports.append({"stem": item.stem,
                             "max_abs_delta": float(delta_effect.max().detach()),
                             "changed_support_pixels": int((delta_effect > 0).any(dim=1).sum().detach())})
        if not torch.isfinite(poison).all():
            raise FloatingPointError("Probe produced non-finite poison input")
    if not finite_loss or not finite_delta or displacement <= 0 or any(x["changed_support_pixels"] <= 0 for x in item_reports):
        raise RuntimeError("Probe failed finite/nonzero perturbation checks")
    if args.method == "entf_push_det" and before != state_after:
        raise RuntimeError("EntF reference state changed during Push")
    output.mkdir(parents=True)
    report = {
        "schema_version": 1, "mode": args.mode, "method": args.method,
        "official_metrics_disabled": True, "artifact_ingest_disabled": True,
        "production_entrypoint_reached": True,
        "official_run": False, "seed": 0, "batch_size": args.batch_size,
        "fixed_train_stems": [x.stem for x in items],
        "dataset_fingerprint": lock["dataset"]["fingerprint"],
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "surrogate_sha256": file_sha256(args.surrogate),
        "candidate_code_commit": args.code_commit,
        "source_file_sha256": source_hashes(Path(__file__).resolve().parent.parent),
        "finite_loss": finite_loss, "loss_observation_count": len(loss_observations),
        "loss_observations": loss_observations, "finite_delta": finite_delta,
        "continued_from_first_block": args.mode == "two-block-smoke",
        "first_block_displacement_mean_abs": float(torch.stack(
            [x.abs().mean() for x in first_block_deltas]).mean().detach()),
        "nonzero_target_displacement_mean_abs": displacement,
        "model_state_before": before, "model_state_after": state_after,
        "surrogate_updated": before != state_after,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "update_duration_seconds": elapsed,
        "index_and_load_duration_seconds": begin_update - start,
        "counters": counters, "per_item": item_reports,
        "cleanup_policy": "retain_all", "checkpoint_cleanup_completed": False,
        "deleted_paths": [], "checkpoint_paths_remaining": [],
        "centroid_identity": locals().get("centroid_identity"),
    }
    (output / "probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if extractor is not None:
        extractor.close()
    print(json.dumps({"state": "passed", "output": str(output / "probe.json"),
                      "duration_seconds": elapsed, "peak_gpu_memory_bytes": report["peak_gpu_memory_bytes"],
                      "counters": counters}, ensure_ascii=False))


if __name__ == "__main__":
    main()
