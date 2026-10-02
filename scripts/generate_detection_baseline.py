"""Generate complete, mechanism-preserving VOC/person poison cycles.

The batch dimension only groups independent sample-wise updates. Official
method-internal loops, losses, radii, sampling and step sizes are fixed here.
This script writes only target-image poison PNGs and small run evidence; full
dataset materialization and victim training are separate audited stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.nms import non_max_suppression

from ue_framework.baselines_det.core import (
    EPS, TARGET_ID, _xyxy, file_sha256, letterbox, model_state_sha256,
)
from ue_framework.baselines_det.production import (
    BatchItem, BatchedPersonROIFeatures, _batch_dict, _loss_on_batch,
    em_noise_batch, em_train_batch, entf_centroid_batch, entf_push_batch,
    make_item, new_counters, rem_defender_batch, sem_defender_batch,
)
from ue_framework.baselines_det.locked_data import locked_source_inventory
from ue_framework.data_utils import (
    label_path_for_image, list_images, read_yolo_annotations,
)


METHODS = ("em_det", "rem_det", "entf_push_det", "sem_det")
ALLOWED_ROOT = Path(
    "/project/guoqinglong/projects/legacy_kproto_ret_dev_20260923/"
    "b2_adaptive_dev_20260926/semantic_dev_20260926/baseline_dev_20260928"
)
IMG_SIZE = 640
SURROGATE_LR = 1e-4
EM_TRAIN_BATCHES = 10
EM_NOISE_STEPS = 20
EM_NOISE_STEP = 0.8 / 255.0
EM_STOP_ERROR = 0.01
DEFENDER_STEPS = 10
DEFENDER_STEP = 1.6 / 255.0
EOT_SAMPLES = 5
INNER_RADIUS = 4.0 / 255.0
INNER_STEPS = 10
INNER_STEP = 0.8 / 255.0
ENTF_STEPS = 300
ENTF_STEP = EPS * 0.05
PERSON_COUNT = 6095
OBJECT_COUNT = 13256
BUDGET_REFERENCE_CYCLE_SECONDS = {
    "em_det": 531.7430627346039,
    "rem_det": 11004.786202669144,
    "entf_push_det": 11301.258711338043,
    "sem_det": 1307.75403881073,
}
BUDGET_CYCLE_FACTOR = 1.25
BUDGET_FINAL_RESERVE_SECONDS = 600.0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


SOURCE_FILES = (
    "scripts/generate_detection_baseline.py",
    "scripts/baseline_batch_probe.py",
    "ue_framework/baselines_det/production.py",
    "ue_framework/baselines_det/core.py",
    "ue_framework/baselines_det/em.py",
    "ue_framework/baselines_det/rem.py",
    "ue_framework/baselines_det/sem.py",
    "ue_framework/baselines_det/entf.py",
    "ue_framework/baselines_det/robust_common.py",
    "ue_framework/baselines_det/locked_data.py",
    "ue_framework/data_utils.py",
)


def source_hashes(root: Path) -> dict:
    return {name: file_sha256(root / name) for name in SOURCE_FILES}


def update_status(path: Path, status: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_item(image_path: Path, labels: Path, device: torch.device) -> BatchItem:
    label_path = Path(label_path_for_image(str(image_path), str(labels)))
    anns = read_yolo_annotations(str(label_path))
    if not any(int(x["cls"]) == TARGET_ID for x in anns):
        raise ValueError(f"Target image set contains a person-free row: {image_path.stem}")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot decode train image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    clean = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(device).float().div_(255.0)
    return make_item(clean, anns, image_path.stem)


def batches(indices: list[int], size: int):
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


def expected_cycle_counters(method: str, batch_count: int) -> dict[str, int]:
    counters = {"forward_calls": 0, "backward_calls": 0, "optimizer_steps": 0,
                "pgd_steps": 0, "noise_steps": 0, "eot_samples": 0,
                "stability_samples": 0, "push_steps": 0}
    if method == "em_det":
        counters.update(forward_calls=EM_TRAIN_BATCHES + EM_NOISE_STEPS * batch_count + PERSON_COUNT,
                        backward_calls=EM_TRAIN_BATCHES + EM_NOISE_STEPS * batch_count,
                        optimizer_steps=EM_TRAIN_BATCHES,
                        noise_steps=PERSON_COUNT * EM_NOISE_STEPS)
    elif method == "rem_det":
        calls_per_batch = DEFENDER_STEPS * EOT_SAMPLES * (INNER_STEPS + 1) + INNER_STEPS + 1
        counters.update(forward_calls=calls_per_batch * batch_count,
                        backward_calls=calls_per_batch * batch_count,
                        optimizer_steps=batch_count,
                        pgd_steps=PERSON_COUNT * (DEFENDER_STEPS * EOT_SAMPLES * INNER_STEPS + INNER_STEPS),
                        eot_samples=PERSON_COUNT * DEFENDER_STEPS * EOT_SAMPLES)
    elif method == "sem_det":
        calls_per_batch = DEFENDER_STEPS * EOT_SAMPLES + INNER_STEPS + 1
        counters.update(forward_calls=calls_per_batch * batch_count,
                        backward_calls=calls_per_batch * batch_count,
                        optimizer_steps=batch_count,
                        pgd_steps=PERSON_COUNT * INNER_STEPS,
                        stability_samples=PERSON_COUNT * DEFENDER_STEPS * EOT_SAMPLES)
    else:
        counters.update(forward_calls=batch_count * (1 + ENTF_STEPS),
                        backward_calls=batch_count * ENTF_STEPS,
                        push_steps=PERSON_COUNT * ENTF_STEPS)
    return counters


def expected_total_counters(method: str, batch_count: int, cycles: int) -> dict[str, int]:
    """A frozen EntF centroid is computed once; all other work repeats."""
    one_cycle = expected_cycle_counters(method, batch_count)
    total = {name: count * cycles for name, count in one_cycle.items()}
    if method == "entf_push_det":
        total["forward_calls"] -= (cycles - 1) * batch_count
    return total


def budget_allows_next_cycle(method: str, deadline: float, now: float,
                             measured_cycle_seconds: list[float]) -> bool:
    reference = max(BUDGET_REFERENCE_CYCLE_SECONDS[method], *measured_cycle_seconds)
    return now + reference * BUDGET_CYCLE_FACTOR + BUDGET_FINAL_RESERVE_SECONDS < deadline


def load_batch(paths: list[Path], labels: Path, device: torch.device) -> list[BatchItem]:
    return [read_item(p, labels, device) for p in paths]


def write_poison(item: BatchItem, delta: torch.Tensor, path: Path) -> None:
    value = (item.clean + (delta.clamp(-EPS, EPS) * item.support)).clamp(0, 1)
    rgb = np.rint(value.detach().cpu()[0].permute(1, 2, 0).numpy() * 255.0)
    rgb = rgb.clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", compress_level=1)


def _train_error_for_batch(model, items, deltas, imgsz: int = IMG_SIZE) -> tuple[int, int]:
    batch_images = [((x.clean + delta * x.support).clamp(0, 1)).detach() for x, delta in zip(items, deltas)]
    packed = _batch_dict(batch_images, [x.annotations for x in items], None, imgsz)
    with torch.no_grad():
        model.eval()
        prediction = model(packed["img"])
        detections = non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.7,
                                         classes=[TARGET_ID], max_det=300)
    missed = total = 0
    for item, det in zip(items, detections):
        _, geometry = letterbox(item.clean, imgsz)
        predicted = det[:, :4]
        for ann in item.annotations:
            if int(ann["cls"]) != TARGET_ID:
                continue
            x1, y1, x2, y2 = _xyxy(ann["bbox"], geometry.original_height, geometry.original_width)
            gt = torch.tensor([[
                x1 * geometry.new_width / geometry.original_width + geometry.pad_left,
                y1 * geometry.new_height / geometry.original_height + geometry.pad_top,
                x2 * geometry.new_width / geometry.original_width + geometry.pad_left,
                y2 * geometry.new_height / geometry.original_height + geometry.pad_top,
            ]], device=item.clean.device, dtype=item.clean.dtype)
            hit = bool(predicted.numel() and (box_iou(gt, predicted) >= 0.5).any())
            total += 1
            missed += int(not hit)
    return missed, total


def _assert_lock(lock: dict, surrogate: Path, data_root: Path) -> tuple[Path, Path, list[Path]]:
    if lock.get("status") != "matched_clean_baseline_complete":
        raise RuntimeError("The protocol lock does not contain a completed matched clean baseline")
    if lock["poison"]["epsilon"] != "8/255" or lock["target"]["class_id"] != TARGET_ID:
        raise RuntimeError("The common epsilon/target protocol differs")
    if lock["clean_baseline"]["status"] != "matched_complete":
        raise RuntimeError("A matched complete clean baseline is required")
    if file_sha256(surrogate) != lock["surrogate"]["checkpoint_sha256"]:
        raise RuntimeError("The clean VOC surrogate checkpoint SHA256 differs from protocol lock")
    inventory, counts = locked_source_inventory(lock, data_root)
    if counts["train_fingerprint"] != lock["dataset"]["train_split_fingerprint"]:
        raise RuntimeError("Recomputed train fingerprint differs from the protocol lock")
    images = Path(data_root) / "images" / "train"
    labels = Path(data_root) / "labels" / "train"
    target_images = []
    labels_by_stem = {path.stem: path for path in inventory["train"]["labels"]}
    for image_path in inventory["train"]["images"]:
        anns = read_yolo_annotations(str(labels_by_stem[image_path.stem]))
        if any(int(x["cls"]) == TARGET_ID for x in anns):
            target_images.append(image_path)
    if len(target_images) != PERSON_COUNT or counts["target_objects"] != OBJECT_COUNT:
        raise RuntimeError(f"Train target index mismatch: images={len(target_images)}, objects={counts['target_objects']}")
    return images, labels, target_images


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--surrogate", type=Path, required=True)
    p.add_argument("--protocol-lock", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--time-cap-hours", type=float, default=8.0)
    p.add_argument("--max-cycles", type=int, default=1)
    p.add_argument("--budget-variant", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--code-commit", default="unspecified")
    p.add_argument("--imgsz", type=int, default=IMG_SIZE)
    args = p.parse_args()
    out_root = args.output_root.resolve()
    try:
        out_root.relative_to(ALLOWED_ROOT)
    except ValueError as exc:
        raise PermissionError(f"Output must remain under authorized root {ALLOWED_ROOT}") from exc
    if out_root.exists():
        raise FileExistsError(f"Refusing to reuse an existing RunID output: {out_root}")
    if args.batch_size < 1 or args.time_cap_hours <= 0 or args.imgsz != IMG_SIZE or args.max_cycles < 1:
        raise ValueError("Invalid batch/time/imgsz setting")
    if args.time_cap_hours > 8.0:
        raise ValueError("Per-method generation cap may not exceed the user-authorized eight hours")
    if args.max_cycles > 1 and not args.budget_variant:
        raise ValueError("Multi-cycle generation must be labeled as the budget variant")
    started = time.time()
    deadline = started + args.time_cap_hours * 3600.0
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    images_dir, labels_dir, target_paths = _assert_lock(lock, args.surrogate, args.dataset_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Official generation requires the authorized CUDA device")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = YOLO(str(args.surrogate)).model.to(device)
    if int(model.nc) != 20:
        raise RuntimeError(f"Expected a VOC20 detector, got nc={model.nc}")
    torch.cuda.reset_peak_memory_stats(device)
    if args.method == "entf_push_det":
        extractor = BatchedPersonROIFeatures(model, imgsz=args.imgsz)
    else:
        extractor = None
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=SURROGATE_LR)

    out_root.mkdir(parents=True)
    outputs = out_root / "noise"
    outputs.mkdir()
    status_path = out_root / "status.json"
    counters = new_counters()
    spec = {
        "schema_version": 1, "run_id": args.run_id, "method": args.method,
        "official_run": True, "state": "running", "stage": "generation",
        "seed": args.seed, "batch_size": args.batch_size, "imgsz": args.imgsz,
        "time_cap_hours": args.time_cap_hours,
        "budget_variant": args.budget_variant, "requested_max_cycles": args.max_cycles,
        "cycle_guard": {"factor": BUDGET_CYCLE_FACTOR,
                        "final_reserve_seconds": BUDGET_FINAL_RESERVE_SECONDS,
                        "reference_cycle_seconds": BUDGET_REFERENCE_CYCLE_SECONDS[args.method]},
        "dataset_fingerprint": lock["dataset"]["fingerprint"],
        "protocol_lock_sha256": file_sha256(args.protocol_lock),
        "surrogate_sha256": file_sha256(args.surrogate),
        "candidate_code_commit": args.code_commit,
        "source_file_sha256": source_hashes(Path(__file__).resolve().parent.parent),
        "train_person_image_count": len(target_paths),
        "train_person_object_count": OBJECT_COUNT,
        "method_schedule": {}, "counters": counters, "cycle_history": [],
        "started_at_unix": started, "completed_at_unix": None,
    }
    (out_root / "run_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    update_status(status_path, spec)
    state_before = model_state_sha256(model)
    max_memory = 0
    person_order = list(range(len(target_paths)))
    rng = np.random.default_rng(args.seed)
    if args.method != "em_det":
        rng.shuffle(person_order)
    if args.method == "em_det":
        spec["method_schedule"] = {
            "complete_outer_cycles": 0,
            "requested_max_complete_outer_cycles": args.max_cycles,
            "surrogate_batches_per_cycle": EM_TRAIN_BATCHES,
            "noise_descent_steps_per_image": EM_NOISE_STEPS,
            "noise_step_size": EM_NOISE_STEP,
            "stop_error_threshold_train_only": EM_STOP_ERROR,
            "stop_policy": "budget_full_cycles" if args.budget_variant else "training_error_or_one_cycle",
        }
    elif args.method == "rem_det":
        spec["method_schedule"] = {
            "complete_outer_cycles": 0,
            "requested_max_complete_outer_cycles": args.max_cycles,
            "defender_steps": DEFENDER_STEPS,
            "eot_samples_per_defender_step": EOT_SAMPLES,
            "adversarial_pgd_steps_per_eot": INNER_STEPS,
            "adversarial_radius": INNER_RADIUS,
            "surrogate_adversarial_pgd_steps": INNER_STEPS,
            "defender_step_size": DEFENDER_STEP, "adversarial_step_size": INNER_STEP,
        }
    elif args.method == "sem_det":
        spec["method_schedule"] = {
            "complete_outer_cycles": 0,
            "requested_max_complete_outer_cycles": args.max_cycles,
            "defender_steps": DEFENDER_STEPS,
            "uniform_stability_samples_per_defender_step": EOT_SAMPLES,
            "uniform_stability_radius": INNER_RADIUS,
            "defender_adversarial_pgd_steps": 0,
            "surrogate_adversarial_pgd_steps": INNER_STEPS,
            "defender_step_size": DEFENDER_STEP, "adversarial_step_size": INNER_STEP,
        }
    else:
        spec["method_schedule"] = {
            "complete_outer_cycles": 0,
            "requested_max_complete_outer_cycles": args.max_cycles,
            "centroid_person_objects": OBJECT_COUNT,
            "push_steps_per_person_image": ENTF_STEPS, "push_step_size": ENTF_STEP,
            "push_init": "uniform[-8/255,8/255] once, then persist delta across cycles",
        }
    batch_count = (PERSON_COUNT + args.batch_size - 1) // args.batch_size
    spec["expected_full_cycle_counters"] = expected_cycle_counters(args.method, batch_count)
    spec["expected_target_batch_count"] = batch_count
    update_status(status_path, spec)
    records_written = 0
    cycles_completed = 0
    cycle_seconds: list[float] = []
    delta_state: dict[str, torch.Tensor] = {}
    train_error = None

    def load_cycle_items(index_batch: list[int], *, continuing: bool) -> list[BatchItem]:
        items = load_batch([target_paths[i] for i in index_batch], labels_dir, device)
        if continuing:
            for item in items:
                if item.stem not in delta_state:
                    raise RuntimeError(f"Missing persistent perturbation for {item.stem}")
                item.raw_delta = delta_state[item.stem].to(device)
        return items

    def retain_deltas(items: list[BatchItem], deltas: list[torch.Tensor]) -> None:
        for item, delta in zip(items, deltas):
            delta_state[item.stem] = delta.detach().cpu().clone()

    try:
        if args.method == "entf_push_det":
            assert extractor is not None
            def centroid_batches():
                for index_batch in batches(person_order, args.batch_size):
                    if time.time() >= deadline:
                        raise TimeoutError("8-hour cap reached during full-train EntF centroid pass")
                    yield load_batch([target_paths[i] for i in index_batch], labels_dir, device)
            centroid, object_count, centroid_batch_count = entf_centroid_batch(extractor, centroid_batches())
            counters["forward_calls"] += centroid_batch_count
            if object_count != OBJECT_COUNT:
                raise RuntimeError(f"EntF centroid used {object_count}, expected {OBJECT_COUNT} person objects")
            spec["centroid_objects_seen"] = object_count
            spec["centroid_batches"] = centroid_batch_count
            update_status(status_path, spec)

        for cycle_index in range(args.max_cycles):
            if cycle_index > 0 and args.method != "em_det":
                rng.shuffle(person_order)
            cycle_started = time.time()
            records_written = 0
            spec.update({"current_cycle": cycle_index + 1,
                         "completed_target_images": 0,
                         "counters": counters})
            update_status(status_path, spec)

            if args.method == "em_det":
                # Preserve train -> noise -> training-error evaluation.
                # The budget variant records the original stop criterion but
                # follows the user-authorized full-cycle wall-clock budget.
                surrogate_indices = list(range(len(target_paths)))
                for batch_index, index_batch in enumerate(batches(surrogate_indices, args.batch_size)):
                    if batch_index >= EM_TRAIN_BATCHES:
                        break
                    if time.time() >= deadline:
                        raise TimeoutError("8-hour cap reached inside an EM cycle")
                    items = load_cycle_items(index_batch, continuing=cycle_index > 0)
                    em_train_batch(model, optimizer, items, imgsz=args.imgsz, counters=counters)
                    spec["surrogate_batch_count_completed"] = counters["optimizer_steps"]
                    spec["counters"] = counters
                    update_status(status_path, spec)
                while counters["optimizer_steps"] < EM_TRAIN_BATCHES * (cycle_index + 1):
                    start = (counters["optimizer_steps"] % EM_TRAIN_BATCHES) * args.batch_size
                    index_batch = list(range(start, min(start + args.batch_size, len(target_paths))))
                    if not index_batch:
                        index_batch = list(range(min(args.batch_size, len(target_paths))))
                    items = load_cycle_items(index_batch, continuing=cycle_index > 0)
                    em_train_batch(model, optimizer, items, imgsz=args.imgsz, counters=counters)
                total_train_missed = total_train_person = 0
                for index_batch in batches(person_order, args.batch_size):
                    if time.time() >= deadline:
                        raise TimeoutError("8-hour cap reached inside an EM noise cycle")
                    items = load_cycle_items(index_batch, continuing=cycle_index > 0)
                    deltas = em_noise_batch(model, items, steps=EM_NOISE_STEPS,
                                            step_size=EM_NOISE_STEP, imgsz=args.imgsz, counters=counters)
                    retain_deltas(items, deltas)
                    for item, delta in zip(items, deltas):
                        write_poison(item, delta, outputs / f"{item.stem}.png")
                        missed, total = _train_error_for_batch(model, [item], [delta], args.imgsz)
                        counters["forward_calls"] += 1
                        total_train_missed += missed
                        total_train_person += total
                        records_written += 1
                    spec.update({"completed_target_images": records_written,
                                 "train_error_person_miss_rate_so_far":
                                     total_train_missed / max(1, total_train_person),
                                 "counters": counters})
                    update_status(status_path, spec)
                    max_memory = max(max_memory, torch.cuda.max_memory_allocated(device))
                train_error = total_train_missed / max(1, total_train_person)
                spec["training_detection_error"] = train_error
                spec["training_error_threshold_met"] = train_error <= EM_STOP_ERROR
                if train_error <= EM_STOP_ERROR:
                    spec.setdefault("training_error_threshold_first_met_cycle", cycle_index + 1)

            elif args.method in ("rem_det", "sem_det"):
                for batch_index, index_batch in enumerate(batches(person_order, args.batch_size)):
                    if time.time() >= deadline:
                        raise TimeoutError("8-hour cap reached inside a defender cycle")
                    items = load_cycle_items(index_batch, continuing=cycle_index > 0)
                    method_seed = args.seed + cycle_index * batch_count + batch_index
                    if args.method == "rem_det":
                        deltas, _ = rem_defender_batch(
                            model, optimizer, items, imgsz=args.imgsz,
                            seed=method_seed, counters=counters)
                    else:
                        deltas, _ = sem_defender_batch(
                            model, optimizer, items, imgsz=args.imgsz,
                            seed=method_seed, counters=counters)
                    retain_deltas(items, deltas)
                    for item, delta in zip(items, deltas):
                        write_poison(item, delta, outputs / f"{item.stem}.png")
                        records_written += 1
                    spec.update({"completed_target_images": records_written, "counters": counters})
                    update_status(status_path, spec)
                    max_memory = max(max_memory, torch.cuda.max_memory_allocated(device))

            else:
                assert extractor is not None
                for batch_index, index_batch in enumerate(batches(person_order, args.batch_size)):
                    if time.time() >= deadline:
                        raise TimeoutError("8-hour cap reached inside an EntF Push cycle")
                    items = load_cycle_items(index_batch, continuing=cycle_index > 0)
                    initial_deltas = [item.raw_delta for item in items] if cycle_index > 0 else None
                    deltas = entf_push_batch(
                        extractor, items, centroid, steps=ENTF_STEPS,
                        step_size=ENTF_STEP, seed=args.seed + batch_index,
                        counters=counters, initial_deltas=initial_deltas)
                    retain_deltas(items, deltas)
                    for item, delta in zip(items, deltas):
                        write_poison(item, delta, outputs / f"{item.stem}.png")
                        records_written += 1
                    spec.update({"completed_target_images": records_written, "counters": counters})
                    update_status(status_path, spec)
                    max_memory = max(max_memory, torch.cuda.max_memory_allocated(device))

            if records_written != PERSON_COUNT or len(delta_state) != PERSON_COUNT:
                raise RuntimeError(
                    f"Incomplete cycle {cycle_index + 1}: images={records_written}, deltas={len(delta_state)}"
                )
            cycles_completed += 1
            cycle_seconds.append(time.time() - cycle_started)
            spec["method_schedule"]["complete_outer_cycles"] = cycles_completed
            spec["cycle_history"].append({
                "cycle": cycles_completed, "duration_seconds": cycle_seconds[-1],
                "completed_target_images": records_written,
                "training_detection_error": train_error,
                "counters": dict(counters),
            })
            spec["counters"] = counters
            update_status(status_path, spec)
            if not args.budget_variant:
                spec["stop_reason"] = ("train_error_threshold"
                                       if args.method == "em_det" and train_error <= EM_STOP_ERROR
                                       else "one_complete_cycle_budget")
                break
            if cycles_completed >= args.max_cycles:
                spec["stop_reason"] = "registered_max_cycles"
                break
            if not budget_allows_next_cycle(args.method, deadline, time.time(), cycle_seconds):
                spec["stop_reason"] = "projected_full_cycle_budget_guard"
                break

        if records_written != PERSON_COUNT:
            raise RuntimeError(f"Incomplete target coverage: {records_written}/{PERSON_COUNT}")
        if time.time() > deadline:
            raise TimeoutError("Completed generation exceeded the eight-hour cap")
        expected_counters = expected_total_counters(args.method, batch_count, cycles_completed)
        spec["expected_total_counters"] = expected_counters
        if counters != expected_counters:
            raise RuntimeError(
                f"Full-cycle counters differ from registered schedule: expected={expected_counters}, actual={counters}"
            )
        state_after = model_state_sha256(model)
        spec.update({"state": "complete", "stage": "generation", "completed_target_images": records_written,
                     "training_detection_error": train_error,
                     "surrogate_updated": state_before != state_after,
                     "surrogate_update_expected": args.method != "entf_push_det",
                     "model_state_before": state_before, "model_state_after": state_after,
                     "peak_gpu_memory_bytes": max_memory,
                     "duration_seconds": time.time() - started,
                     "completed_at_unix": time.time(), "counters": counters,
                     "epsilon": EPS, "overlap_epsilon": EPS / 2,
                     "outside_target_epsilon": 0})
        if spec["surrogate_updated"] != spec["surrogate_update_expected"]:
            raise RuntimeError("Surrogate update behavior differs from the registered method")
        (out_root / "run_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        update_status(status_path, spec)
    except Exception as exc:
        spec.update({"state": "failed", "stage": "generation", "error": repr(exc),
                     "completed_target_images": records_written,
                     "duration_seconds": time.time() - started,
                     "completed_at_unix": time.time(), "counters": counters})
        (out_root / "run_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        update_status(status_path, spec)
        raise
    finally:
        if extractor is not None:
            extractor.close()
    print(json.dumps({"state": "complete", "run_id": args.run_id,
                      "method": args.method, "target_images": records_written,
                      "duration_seconds": spec["duration_seconds"],
                      "training_detection_error": train_error,
                      "peak_gpu_memory_bytes": max_memory,
                      "counters": counters}, ensure_ascii=False))


if __name__ == "__main__":
    main()
