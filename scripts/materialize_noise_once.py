"""Materialize the completed universal perturbation without legacy reset paths."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from ue_framework.config import load_config
from ue_framework.methods.semantic_coordinates import semantic_identity
from ue_framework.data_utils import list_images
from ue_framework.paths import build_run_paths, ensure_run_dirs
from ue_framework.runtime import RunContext


ROOT = Path(__file__).resolve().parent.parent
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--noise", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    noise = Path(args.noise).resolve()
    if not noise.is_relative_to(ROOT.resolve()):
        raise ValueError("Noise run must be inside the isolated project")
    with open(noise / "status.json", encoding="utf-8") as handle:
        status = json.load(handle)
    if status["state"] != "complete" or status.get("smoke"):
        raise RuntimeError("Official noise run has not completed")
    if args.seed != int(status["seed"]):
        raise RuntimeError("Materialization seed differs from the optimized-noise seed")
    if sha256(args.config) != status["config_sha256"]:
        raise RuntimeError("Materialization config differs from the optimized-noise config")
    method_source = ROOT / "ue_framework" / "methods" / "legacy_kproto_ret.py"
    if sha256(method_source) != status["code_sha256"]:
        raise RuntimeError("Materialization method differs from the optimized-noise method")
    support_parent = ROOT / "ue_framework" / "methods" / "tausb_universal.py"
    if sha256(support_parent) != status["support_parent_sha256"]:
        raise RuntimeError("Materialization support implementation differs from noise optimization")
    source = noise / "global_params.pt"
    if not source.is_file():
        raise FileNotFoundError(source)
    cfg = load_config(args.config)
    if semantic_identity(cfg, ROOT) != status.get("semantic_identity"):
        raise RuntimeError("Semantic code/artifact identity differs from optimized noise")
    adaptive_enabled = bool(cfg["methods"]["legacy_kproto_ret"].get("adaptive", {}).get("enabled", False))
    if adaptive_enabled != bool(status.get("adaptive_enabled", False)):
        raise RuntimeError("Adaptive method identity differs from noise optimization")
    if adaptive_enabled:
        for name, key in (("adaptive_learner.py", "adaptive_code_sha256"),
                          ("b2_adaptive.py", "adaptive_trainer_sha256")):
            if status.get(key) != sha256(ROOT / "ue_framework" / "methods" / name):
                raise RuntimeError(f"Adaptive method code differs: {name}")
    dataset = Path(cfg["data"]["dataset_root"])
    originals = list_images(str(dataset / cfg["data"]["train_images"]))
    if len({Path(image).stem for image in originals}) != len(originals):
        raise RuntimeError("Training image stems are not unique")
    for image in originals:
        label = dataset / cfg["data"]["train_labels"] / (Path(image).stem + ".txt")
        if not label.is_file():
            raise FileNotFoundError(label)
    cfg["platform"]["resume"] = True
    epochs = int(cfg["methods"]["legacy_kproto_ret"]["universal_epochs"])
    if cfg["experiment"]["steps"] != [epochs]:
        raise ValueError("B2 steps and universal_epochs must match")
    paths = build_run_paths(cfg["platform"]["run_root"], "legacy_kproto_ret", epochs, args.seed)
    if not Path(paths.run_root).resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Materialization outputs must be inside the isolated project")
    ensure_run_dirs(paths)
    identity_path = Path(paths.poisoned_root) / "materialization_identity.json"
    poisoned_status_path = Path(paths.poisoned_status_json)
    if poisoned_status_path.is_file():
        with open(poisoned_status_path, encoding="utf-8") as handle:
            previous_status = json.load(handle)
        if "generate_poisoned_dataset" in previous_status.get("completed_stages", []) and not identity_path.is_file():
            raise RuntimeError("Completed materialization has no noise-run identity")
    dest = Path(paths.noise_dir) / "global_params.pt"
    if dest.exists():
        if sha256(dest) != sha256(source):
            raise RuntimeError("An existing materialization uses different noise parameters")
    else:
        shutil.copy2(source, dest)
    ctx = RunContext(cfg=cfg, method="legacy_kproto_ret", steps=epochs, seed=args.seed,
                     stage="generate_poisoned_dataset", gpu_id=0, platform_mode="cloud", paths=paths)
    from ue_framework.stages.generate import run_generate_poisoned_dataset
    run_generate_poisoned_dataset(ctx)
    identity = {
        "noise_run": str(noise), "semantic_identity": status.get("semantic_identity"), "noise_global_params_sha256": sha256(source),
        "noise_config_sha256": status["config_sha256"],
        "noise_method_sha256": status["code_sha256"],
        "noise_support_parent_sha256": status["support_parent_sha256"],
        "noise_adaptive_code_sha256": status.get("adaptive_code_sha256"),
        "noise_adaptive_trainer_sha256": status.get("adaptive_trainer_sha256"),
        "materializer_sha256": sha256(__file__),
        "generator_stage_sha256": sha256(ROOT / "ue_framework" / "stages" / "generate.py"),
        "seed": args.seed,
        "source_stems_sha256": hashlib.sha256(
            "\n".join(sorted(Path(image).stem for image in originals)).encode("utf-8")
        ).hexdigest(),
        "source_image_count": len(originals),
    }
    if identity_path.is_file():
        with open(identity_path, encoding="utf-8") as handle:
            previous_identity = json.load(handle)
        if previous_identity != identity:
            raise RuntimeError("Existing materialization identity differs from this noise run")
    else:
        identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    print(json.dumps({"poisoned_root": paths.poisoned_root, "manifest": paths.manifest_csv,
                      "noise_sha256": sha256(source)}, indent=2))


if __name__ == "__main__":
    main()
