from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F
import yaml

from ue_framework.methods.p1_determinism_audit import enable_strict_determinism
from ue_framework.methods.sdh_experiment import (
    _batches,
    _person_paths,
    _resolve,
    _split_hash,
    deterministic_person_split,
    validate_sdh_experiment_config,
)
from ue_framework.methods.sdh_mechanism import load_sdh_batch
from ue_framework.methods.semantic_hiding_carrier import (
    deterministic_bilinear_resize_2d,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe deterministic SDH box-resize forward and backward."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=32)
    return parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _first_batch_sizes(config: dict, *, config_base: Path) -> tuple[list[list[int]], str]:
    dataset_root = _resolve(config_base, str(config["dataset"]["root"]))
    image_dir = dataset_root / str(config["dataset"]["train_images"])
    label_dir = dataset_root / str(config["dataset"]["train_labels"])
    paths = _person_paths(image_dir, label_dir, 14)
    split = deterministic_person_split(
        paths,
        label_dir=label_dir,
        target_class_id=14,
        calibration_count=int(config["mechanism"]["calibration_batches"])
        * int(config["mechanism"]["batch_size"]),
        heldout_count=int(config["mechanism"]["heldout_batches"])
        * int(config["mechanism"]["batch_size"]),
        seed=0,
    )
    split_hash = _split_hash(split)
    if split_hash != str(config["dgcaip"]["expected_split_sha256"]):
        raise ValueError("Resize probe split hash mismatch.")
    first_paths = _batches(
        split["calibration"], int(config["mechanism"]["batch_size"])
    )[0]
    batch = load_sdh_batch(
        first_paths,
        label_dir=label_dir,
        image_size=640,
        target_class_id=14,
        device=torch.device("cpu"),
    )
    sizes = []
    for boxes in batch.boxes_by_image:
        for box in boxes:
            left = max(0, min(640, int(math.floor(float(box[0])))))
            top = max(0, min(640, int(math.floor(float(box[1])))))
            right = max(0, min(640, int(math.ceil(float(box[2])))))
            bottom = max(0, min(640, int(math.ceil(float(box[3])))))
            if right > left and bottom > top:
                sizes.append([bottom - top, right - left])
    if not sizes:
        raise ValueError("Resize probe first batch has no valid person boxes.")
    return sizes, split_hash


def _one_repeat(
    source: torch.Tensor,
    probes: list[torch.Tensor],
    sizes: list[list[int]],
) -> tuple[list[str], str, float]:
    current = source.detach().clone().requires_grad_(True)
    outputs = []
    loss = current.new_zeros(())
    max_forward_error = 0.0
    for size, probe in zip(sizes, probes):
        output = deterministic_bilinear_resize_2d(current, (size[0], size[1]))
        with torch.no_grad():
            reference = F.interpolate(
                current.detach(),
                size=(size[0], size[1]),
                mode="bilinear",
                align_corners=False,
            )
            max_forward_error = max(
                max_forward_error,
                float((output.detach() - reference).abs().max()),
            )
        outputs.append(_tensor_sha256(output))
        loss = loss + (output * probe).mean()
    gradient = torch.autograd.grad(loss, current)[0]
    if not torch.isfinite(gradient).all():
        raise ValueError("Resize probe produced a non-finite gradient.")
    return outputs, _tensor_sha256(gradient), max_forward_error


def main() -> int:
    args = _arguments()
    output_path = Path(args.output).resolve()
    started = time.time()
    try:
        if args.iterations != 32:
            raise ValueError("Resize probe iterations must remain 32.")
        config_path = Path(args.config).resolve()
        project_root = Path.cwd().resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        validate_sdh_experiment_config(config)
        sizes, split_hash = _first_batch_sizes(config, config_base=project_root)
        backend = enable_strict_determinism()
        device = torch.device(str(config["runtime"]["device"]))
        torch.manual_seed(0)
        source = torch.randn(1, 3, 256, 256, device=device)
        probes = [
            torch.randn(1, 3, height, width, device=device)
            for height, width in sizes
        ]
        repeats = [_one_repeat(source, probes, sizes) for _ in range(3)]
        forward_hashes = [value[0] for value in repeats]
        gradient_hashes = [value[1] for value in repeats]
        max_forward_error = max(value[2] for value in repeats)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        benchmark_started = time.perf_counter()
        for index in range(args.iterations):
            size = sizes[index % len(sizes)]
            current = source.detach().clone().requires_grad_(True)
            resized = deterministic_bilinear_resize_2d(current, (size[0], size[1]))
            torch.autograd.grad(resized.square().mean(), current)
        torch.cuda.synchronize(device)
        benchmark_seconds = time.perf_counter() - benchmark_started
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        exact = forward_hashes[0] == forward_hashes[1] == forward_hashes[2]
        exact = exact and gradient_hashes[0] == gradient_hashes[1] == gradient_hashes[2]
        passed = bool(exact and max_forward_error <= 2.0e-6)
        result = {
            "schema": "tausb.sdh-deterministic-resize-probe.v1",
            "spec_id": config["spec"]["spec_id"],
            "status": "passed" if passed else "failed",
            "split_sha256": split_hash,
            "person_box_sizes": sizes,
            "repeat_forward_sha256": forward_hashes,
            "repeat_gradient_sha256": gradient_hashes,
            "bitwise_exact": exact,
            "max_forward_abs_error": max_forward_error,
            "benchmark_iterations": args.iterations,
            "benchmark_seconds": benchmark_seconds,
            "peak_cuda_memory_bytes": peak_bytes,
            "backend": backend,
            "started_unix": started,
            "ended_unix": time.time(),
        }
        _write_json(output_path, result)
        return 0 if passed else 1
    except Exception as error:
        _write_json(
            output_path,
            {
                "schema": "tausb.sdh-deterministic-resize-probe.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "started_unix": started,
                "ended_unix": time.time(),
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
