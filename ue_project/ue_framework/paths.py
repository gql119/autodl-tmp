import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class RunPaths:
    run_root: str
    method: str
    steps: int
    seed: int
    run_tag: str
    poisoned_root: str
    artifact_root: str
    poisoned_images: str
    poisoned_labels: str
    manifest_csv: str
    poisoned_status_json: str
    artifact_status_json: str
    noise_dir: str
    logs_dir: str
    metrics_dir: str
    checkpoints_dir: str
    viz_dir: str
    eval_dir: str
    train_project_dir: str
    bundle_dir: str
    bundle_path: str



def _tagged_seed(seed: int, run_tag: str) -> str:
    if not run_tag:
        return f"seed{seed}"
    if os.path.sep in run_tag or (os.path.altsep and os.path.altsep in run_tag):
        raise ValueError(f"run_tag must not contain path separators: {run_tag}")
    return f"seed{seed}_{run_tag}"


def build_run_paths(run_root: str, method: str, steps: int, seed: int, run_tag: str = "") -> RunPaths:
    poisoned_root = os.path.join(run_root, "poisoned_datasets", method, f"steps{steps}", f"seed{seed}")
    artifact_root = os.path.join(run_root, "artifacts", method, f"steps{steps}", _tagged_seed(seed, run_tag))

    bundle_dir = os.path.join(run_root, "bundles")
    bundle_suffix = f"_seed{seed}_{run_tag}" if run_tag else f"_seed{seed}"
    bundle_name = f"{method}_steps{steps}{bundle_suffix}_compact.zip"

    return RunPaths(
        run_root=run_root,
        method=method,
        steps=steps,
        seed=seed,
        run_tag=run_tag,
        poisoned_root=poisoned_root,
        artifact_root=artifact_root,
        poisoned_images=os.path.join(poisoned_root, "images", "train"),
        poisoned_labels=os.path.join(poisoned_root, "labels", "train"),
        manifest_csv=os.path.join(poisoned_root, "manifest.csv"),
        poisoned_status_json=os.path.join(poisoned_root, "status.json"),
        artifact_status_json=os.path.join(artifact_root, "status.json"),
        noise_dir=os.path.join(artifact_root, "noise"),
        logs_dir=os.path.join(artifact_root, "logs"),
        metrics_dir=os.path.join(artifact_root, "metrics"),
        checkpoints_dir=os.path.join(artifact_root, "checkpoints"),
        viz_dir=os.path.join(artifact_root, "viz"),
        eval_dir=os.path.join(artifact_root, "eval"),
        train_project_dir=os.path.join(artifact_root, "train_runs"),
        bundle_dir=bundle_dir,
        bundle_path=os.path.join(bundle_dir, bundle_name),
    )


def apply_poisoned_root_override(paths: RunPaths, poisoned_root: str) -> RunPaths:
    paths.poisoned_root = poisoned_root
    paths.poisoned_images = os.path.join(poisoned_root, "images", "train")
    paths.poisoned_labels = os.path.join(poisoned_root, "labels", "train")
    paths.manifest_csv = os.path.join(poisoned_root, "manifest.csv")
    paths.poisoned_status_json = os.path.join(poisoned_root, "status.json")
    return paths



def ensure_run_dirs(paths: RunPaths) -> None:
    for p in [
        paths.poisoned_images,
        paths.poisoned_labels,
        paths.noise_dir,
        paths.logs_dir,
        paths.metrics_dir,
        paths.checkpoints_dir,
        paths.viz_dir,
        paths.eval_dir,
        paths.train_project_dir,
        paths.bundle_dir,
    ]:
        os.makedirs(p, exist_ok=True)



def to_dict(paths: RunPaths) -> Dict[str, str]:
    return paths.__dict__.copy()

