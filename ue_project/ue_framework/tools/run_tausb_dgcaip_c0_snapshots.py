from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, Mapping, Sequence

from ue_framework.config import load_config
from ue_framework.io_utils import atomic_write_json, read_csv_rows
from ue_framework.paths import apply_poisoned_root_override, build_run_paths
from ue_framework.sparse_dataset import audit_sparse_training_list
from ue_framework.tools.run_tausb_sdh_e2e_v0_oneboot import (
    GuardFailure,
    run_guarded,
)


SPEC_ID = "TAUSB-SDH-DGCAIP-DATASET-CGR-PROXY-v1"
RUN_TAG = "C0"
METHOD = "tausb_sdh"
STEPS = 40
SEED = 0
WALL_SECONDS = 45 * 60
FIRST_PROGRESS_SECONDS = 5 * 60
IDLE_SECONDS = 15 * 60
MIN_FREE_BYTES = 5 * 1024 ** 3
EXPECTED_VICTIM_INIT_TENSOR_SHA256 = (
    "54aaf431f8a67b3f3067319a8164d1d6db6874497a46109a17af76a15d2b994c"
)
EXPECTED_SURROGATE_SHA256 = (
    "8de8a0c78c6414ad0bf98052b3bc96c33d8e854a2a2a905d47c8195363975b89"
)
EXPECTED_VICTIM_INIT_YAML_SHA256 = (
    "b812e8de7c596779f7cb30c2c57953d2c765985af5152204dad9d5c882ecde3e"
)
EXPECTED_TRAIN_IMAGE_MANIFEST_SHA256 = (
    "4954727df8686532a788668fd815092112ac3e3ee1414eba83b616e683708fbd"
)
EXPECTED_SPARSE_REPORT_SHA256 = (
    "e017b0fc06aa899ae62ecdc4f03b3e5e7a3ef32d2024abfd26b572a096519fdf"
)
EXPECTED_SPARSE_FIELDS = {
    "schema": "tausb.sdh-sparse-train-audit.v1",
    "train_list_sha256": (
        "4d069fb25fde7e63fce7426ed2417ff11b0bf1e97f40c489d9d1a63c141ddcd2"
    ),
    "ordered_stems_sha256": (
        "2e0f30546c8848b7f2b9c4239b49ff417dba3d51e2bd54b354fb0f299ea00011"
    ),
    "label_content_manifest_sha256": (
        "022fbdace84899bf5d340cd07f2eb1d51834c3d0bd35b446f4fef11eb2a53216"
    ),
    "total_count": 16551,
    "target_count": 6095,
    "poisoned_count": 0,
    "poisoned_png_count": 0,
    "original_jpeg_count": 16551,
    "materialization_layout": "sparse_mixed_list_v1",
}
SNAPSHOT_EPOCHS = {"e1": 0, "e5": 4, "e20": 19}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh a clean VOC C0 victim once and bind exact post-epoch "
            "1/5/20 checkpoints for the DG-CAIP dataset-risk gate."
        )
    )
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--poisoned-root", required=True)
    parser.add_argument("--required-storage-root", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--tmp-root", required=True)
    parser.add_argument("--python-bin", default="/root/miniconda3/bin/python")
    parser.add_argument("--device", default="0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", default="")
    parser.add_argument("--shutdown-on-exit", action="store_true")
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_size_manifest_sha256(paths: Sequence[Path]) -> str:
    records = [
        {"name": path.name, "size": path.stat().st_size}
        for path in sorted(paths, key=lambda item: item.name)
    ]
    return _canonical_json_sha256(records)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Required JSON is missing: %s" % path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object: %s" % path)
    return payload


def _git(repository_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=str(repository_root), text=True
    ).strip()


def _inside(root: Path, candidate: Path) -> bool:
    root_text = str(root.resolve())
    candidate_text = str(candidate.resolve())
    return os.path.commonpath((root_text, candidate_text)) == root_text


def _require_fresh(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Fresh C0 snapshot paths already exist: %s" % existing)


def _validate_config(config_path: Path) -> Dict[str, Any]:
    cfg = load_config(str(config_path))
    expected = {
        ("experiment", "pilot_kind"): "e20",
        ("experiment", "arm_id"): "C0",
        ("experiment", "poisoning_ratio"): 0.0,
        ("experiment", "expected_poisoned_count"): 0,
        ("experiment", "expected_train_images"): 16551,
        ("experiment", "expected_target_images"): 6095,
        ("data", "materialization_layout"): "sparse_mixed_list_v1",
        ("platform", "save_every_n_epochs"): 1,
        ("platform", "pack_every_n_epochs"): 20,
        ("platform", "zip_after_stage"): False,
        ("victim", "epochs"): 20,
        ("victim", "batch"): 36,
        ("victim", "imgsz"): 640,
        ("victim", "optimizer"): "SGD",
    }
    for (section, key), value in expected.items():
        if cfg[section].get(key) != value:
            raise ValueError("Config %s.%s must be %r." % (section, key, value))
    if cfg["methods"][METHOD].get("protocol_id") != "TAUSB-SDH-E2E-V0-MAP50-v1":
        raise ValueError("C0 refresh must retain the reviewed E2E V0 protocol.")
    return cfg


def _validate_sparse_input(poisoned_root: Path) -> Dict[str, Any]:
    report_path = poisoned_root / "sparse_materialization.json"
    train_list = poisoned_root / "train-images.txt"
    manifest_csv = poisoned_root / "manifest.csv"
    for path in (
        poisoned_root / "images/train",
        poisoned_root / "labels/train",
        report_path,
        train_list,
        manifest_csv,
    ):
        if not path.exists():
            raise FileNotFoundError("Clean sparse input is incomplete: %s" % path)
    if _file_sha256(report_path) != EXPECTED_SPARSE_REPORT_SHA256:
        raise ValueError("Clean sparse materialization report hash differs.")
    report = _read_json(report_path)
    for key, expected in EXPECTED_SPARSE_FIELDS.items():
        if report.get(key) != expected:
            raise ValueError("Sparse report %s differs from reviewed input." % key)
    audit = audit_sparse_training_list(
        str(train_list),
        read_csv_rows(str(manifest_csv)),
        expected_total=16551,
        expected_poisoned=0,
        expected_target=6095,
        target_class_id=14,
        num_classes=20,
    )
    for key, expected in EXPECTED_SPARSE_FIELDS.items():
        if key in audit and audit.get(key) != expected:
            raise ValueError("Live sparse audit %s differs from reviewed input." % key)
    return {
        "root": str(poisoned_root),
        "report_sha256": EXPECTED_SPARSE_REPORT_SHA256,
        "train_list_sha256": report["train_list_sha256"],
        "ordered_stems_sha256": report["ordered_stems_sha256"],
        "label_content_manifest_sha256": report[
            "label_content_manifest_sha256"
        ],
        "total_count": audit["total_count"],
        "target_count": audit["target_count"],
        "poisoned_count": audit["poisoned_count"],
    }


def _gpu_precheck() -> Dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        raise FileNotFoundError("nvidia-smi is unavailable.")
    gpu = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ),
        text=True,
    ).strip()
    if not gpu:
        raise ValueError("No GPU is visible.")
    active = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
        text=True,
    ).strip()
    if active:
        raise ValueError("Another GPU compute process is already active: %s" % active)
    return {"gpu_rows": gpu.splitlines(), "active_compute_processes": []}


def _preflight(args: argparse.Namespace, *, require_gpu: bool) -> Dict[str, Any]:
    repository_root = Path(args.repository_root).resolve()
    project_root = repository_root / "ue_project"
    config_path = Path(args.config).resolve()
    poisoned_root = Path(args.poisoned_root).resolve()
    storage_root = Path(args.required_storage_root).resolve()
    control_root = Path(args.control_root).resolve()
    log_root = Path(args.log_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    tmp_root = Path(args.tmp_root).resolve()
    python_bin = Path(args.python_bin).resolve()

    if _git(repository_root, "rev-parse", "HEAD") != args.expected_commit:
        raise ValueError("Execution checkout does not match expected commit.")
    if _git(
        repository_root, "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("Execution checkout has tracked changes.")
    if not python_bin.is_file():
        raise FileNotFoundError("Python executable is missing: %s" % python_bin)
    if not storage_root.is_dir():
        raise FileNotFoundError("Required storage root is missing: %s" % storage_root)
    for output in (control_root, log_root, cache_root, tmp_root):
        if not _inside(storage_root, output):
            raise ValueError("Output escapes required storage root: %s" % output)

    cfg = _validate_config(config_path)
    run_root = Path(cfg["platform"]["run_root"]).resolve()
    if not _inside(storage_root, run_root):
        raise ValueError("Configured run root escapes required storage root.")
    _require_fresh((run_root, control_root, log_root, cache_root, tmp_root))
    if shutil.disk_usage(str(storage_root)).free < MIN_FREE_BYTES:
        raise ValueError("Data disk has less than 5 GiB free.")

    dataset_root = Path(cfg["data"]["dataset_root"])
    for relative in ("images/train", "labels/train", "images/val", "labels/val"):
        if not (dataset_root / relative).is_dir():
            raise FileNotFoundError("VOC directory is missing: %s" % (dataset_root / relative))
    train_images = list((dataset_root / "images/train").glob("*.jpg"))
    if len(train_images) != 16551:
        raise ValueError("VOC train image count is not 16551.")
    if _path_size_manifest_sha256(train_images) != EXPECTED_TRAIN_IMAGE_MANIFEST_SHA256:
        raise ValueError("VOC train image path/size manifest differs.")
    surrogate = Path(cfg["surrogate"]["ckpt"])
    if _file_sha256(surrogate) != EXPECTED_SURROGATE_SHA256:
        raise ValueError("Surrogate checkpoint hash differs from the reviewed input.")
    victim_init = project_root / cfg["victim"]["init"]
    if not victim_init.is_file():
        raise FileNotFoundError("Victim model YAML is missing: %s" % victim_init)
    if _file_sha256(victim_init) != EXPECTED_VICTIM_INIT_YAML_SHA256:
        raise ValueError("Victim model YAML hash differs from the reviewed input.")

    result = {
        "schema": "tausb.dgcaip-c0-snapshot-preflight.v1",
        "spec_id": SPEC_ID,
        "status": "passed",
        "execution_commit": args.expected_commit,
        "repository_root": str(repository_root),
        "config": str(config_path),
        "config_sha256": _file_sha256(config_path),
        "run_root": str(run_root),
        "storage_root": str(storage_root),
        "storage_free_bytes": shutil.disk_usage(str(storage_root)).free,
        "victim_init": str(victim_init),
        "victim_init_yaml_sha256": _file_sha256(victim_init),
        "expected_victim_init_tensor_sha256": EXPECTED_VICTIM_INIT_TENSOR_SHA256,
        "surrogate_checkpoint_sha256": EXPECTED_SURROGATE_SHA256,
        "sparse_input": _validate_sparse_input(poisoned_root),
        "gpu_required": bool(require_gpu),
    }
    if require_gpu:
        result.update(_gpu_precheck())
    return result


def validate_checkpoint_epoch(payload: Mapping[str, Any], expected_epoch: int) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload is not a mapping.")
    actual = payload.get("epoch")
    if isinstance(actual, bool) or not isinstance(actual, int):
        raise ValueError("Checkpoint epoch metadata is not an integer.")
    if actual != expected_epoch:
        raise ValueError(
            "Checkpoint epoch metadata is %r, expected %d." % (actual, expected_epoch)
        )


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload is not a mapping: %s" % path)
    return payload


def _verify_result(args: argparse.Namespace, preflight: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = load_config(args.config)
    paths = build_run_paths(
        cfg["platform"]["run_root"], METHOD, STEPS, SEED, run_tag=RUN_TAG
    )
    paths = apply_poisoned_root_override(paths, args.poisoned_root)
    status = _read_json(Path(paths.artifact_status_json))
    if "train_victim" not in set(status.get("completed_stages", [])):
        raise ValueError("C0 train_victim stage is not completed.")
    train_state = status.get("stage_state", {}).get("train_victim", {})
    if int(train_state.get("latest_epoch", -1)) < 19:
        raise ValueError("C0 victim did not finish 20 epochs.")

    fresh_init_path = Path(paths.logs_dir) / "fresh_init.json"
    fresh_init = _read_json(fresh_init_path)
    if fresh_init.get("resume_enabled") is not False:
        raise ValueError("C0 victim was not a fresh initialization.")
    if fresh_init.get("victim_init_tensor_sha256") != EXPECTED_VICTIM_INIT_TENSOR_SHA256:
        raise ValueError("C0 fresh initialization tensor hash differs.")
    if fresh_init.get("matches_expected_victim_init") is not True:
        raise ValueError("C0 fresh initialization binding did not pass.")

    weights_root = Path(paths.train_project_dir) / "victim/weights"
    snapshots: Dict[str, Dict[str, Any]] = {}
    for snapshot_id, epoch_index in SNAPSHOT_EPOCHS.items():
        checkpoint = weights_root / ("epoch%d.pt" % epoch_index)
        if not checkpoint.is_file():
            raise FileNotFoundError("Required snapshot is missing: %s" % checkpoint)
        validate_checkpoint_epoch(_load_checkpoint(checkpoint), epoch_index)
        snapshots[snapshot_id] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _file_sha256(checkpoint),
            "checkpoint_epoch_index": epoch_index,
            "epochs_completed": epoch_index + 1,
        }

    return {
        "schema": "tausb.dgcaip-c0-snapshot-manifest.v1",
        "spec_id": SPEC_ID,
        "status": "passed",
        "execution_commit": args.expected_commit,
        "preflight": dict(preflight),
        "fresh_init": fresh_init,
        "artifact_status": paths.artifact_status_json,
        "snapshots": snapshots,
        "created_unix": time.time(),
    }


def _launch_command(args: argparse.Namespace) -> Sequence[str]:
    return (
        str(Path(args.python_bin).resolve()),
        "-u",
        "-m",
        "ue_framework.launch_one",
        "--config",
        str(Path(args.config).resolve()),
        "--method",
        METHOD,
        "--steps",
        str(STEPS),
        "--seed",
        str(SEED),
        "--stage",
        "train_victim",
        "--gpu_id",
        str(args.device),
        "--poisoned_root_override",
        str(Path(args.poisoned_root).resolve()),
        "--run_tag",
        RUN_TAG,
    )


def _run(args: argparse.Namespace) -> int:
    preflight = _preflight(args, require_gpu=not args.preflight_only)
    if args.preflight_only:
        if args.preflight_output:
            output = Path(args.preflight_output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(str(output), preflight)
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    repository_root = Path(args.repository_root).resolve()
    project_root = repository_root / "ue_project"
    control_root = Path(args.control_root).resolve()
    log_root = Path(args.log_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    tmp_root = Path(args.tmp_root).resolve()
    control_root.mkdir(parents=True, exist_ok=False)
    log_root.mkdir(parents=True, exist_ok=False)
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    status_path = control_root / "controller_status.json"
    status: Dict[str, Any] = {
        "schema": "tausb.dgcaip-c0-snapshot-controller-status.v1",
        "spec_id": SPEC_ID,
        "status": "running",
        "current_stage": "C0_TRAIN",
        "execution_commit": args.expected_commit,
        "hard_cap_seconds": WALL_SECONDS,
        "started_unix": time.time(),
    }
    atomic_write_json(str(status_path), status)
    atomic_write_json(str(control_root / "preflight.json"), preflight)
    os.environ.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TMPDIR": str(tmp_root),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "TORCH_HOME": str(cache_root / "torch"),
            "YOLO_CONFIG_DIR": str(cache_root / "yolo"),
            "TAUSB_EXPECTED_VICTIM_INIT_TENSOR_SHA256": (
                EXPECTED_VICTIM_INIT_TENSOR_SHA256
            ),
        }
    )
    try:
        train = run_guarded(
            stage="C0_TRAIN",
            command=_launch_command(args),
            cwd=project_root,
            log_path=log_root / "c0_train.log",
            wall_seconds=WALL_SECONDS,
            first_progress_seconds=FIRST_PROGRESS_SECONDS,
            idle_seconds=IDLE_SECONDS,
            require_gpu=True,
        )
        status["current_stage"] = "SNAPSHOT_VERIFY"
        status["train"] = train
        atomic_write_json(str(status_path), status)
        manifest = _verify_result(args, preflight)
        manifest_path = control_root / "c0_snapshot_manifest.json"
        atomic_write_json(str(manifest_path), manifest)
        status.update(
            {
                "status": "completed",
                "current_stage": "",
                "snapshot_manifest": str(manifest_path),
                "ended_unix": time.time(),
            }
        )
        atomic_write_json(str(status_path), status)
        return 0
    except BaseException as error:
        status.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "ended_unix": time.time(),
            }
        )
        atomic_write_json(str(status_path), status)
        raise


def _shutdown() -> None:
    shutdown = Path("/usr/bin/shutdown")
    if shutdown.is_file():
        subprocess.run((str(shutdown), "-h", "now"), check=False)


def main() -> int:
    args = _arguments()
    should_shutdown = bool(args.shutdown_on_exit and not args.preflight_only)
    try:
        return _run(args)
    except BaseException as error:
        print(
            "[DGCAIP-C0-Snapshots] %s: %s" % (type(error).__name__, error),
            file=sys.stderr,
        )
        return int(getattr(error, "exit_code", 1))
    finally:
        if should_shutdown:
            _shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
