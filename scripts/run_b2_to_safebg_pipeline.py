"""B2-TO-Safe: build frozen coordinates, sanity gate, then full run if admitted."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Pipeline path outside isolated project: {resolved}")
    return resolved


def reference_read_path(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT.resolve().parents[1]):
        raise ValueError(f"Reference metric is outside approved semantic_dev tree: {resolved}")
    return resolved


def validate_admission(config, admission):
    receipt = json.loads(admission.read_text(encoding="utf-8"))
    if (receipt.get("sanity_gate") != "passed"
            or receipt.get("scientific_review") != "scientifically_correct"):
        raise RuntimeError("B2-TO-Safe has not passed both required admission gates")
    if receipt.get("config_sha256") != sha256(config):
        raise RuntimeError("Admission config identity differs")
    for path_key, hash_key, state_key in (
            ("config_audit_path", "config_audit_sha256", "config_audit_state"),
            ("sanity_gate_path", "sanity_gate_sha256", "sanity_gate_state"),
            ("scientific_review_path", "scientific_review_sha256", "scientific_review")):
        evidence_path = inside(receipt[path_key])
        if sha256(evidence_path) != receipt[hash_key]:
            raise RuntimeError(f"Admission evidence changed: {path_key}")
        if state_key == "scientific_review":
            verdict = json.loads(evidence_path.read_text(encoding="utf-8"))
            if verdict.get("verdict") != "scientifically_correct":
                raise RuntimeError("Scientific review verdict does not approve this candidate")
        elif receipt.get(state_key) != "passed":
            raise RuntimeError(f"Admission evidence did not pass: {path_key}")
    required = {
        "ue_framework/methods/b2_semantic.py", "ue_framework/methods/safebg.py",
        "ue_framework/methods/semantic_coordinates.py", "ue_framework/methods/orth_reference_design.py",
        "ue_framework/methods/b2_adaptive.py", "ue_framework/methods/adaptive_learner.py",
        "scripts/build_b2_to_safe_coordinates.py", "scripts/train_noise_once.py",
        "scripts/materialize_noise_once.py", "scripts/run_isolated_victim.py",
        "scripts/audit_b2_to_safebg_materialization.py", "scripts/check_b2_to_safebg_sanity.py",
        "scripts/evaluate_b2_to_safebg_checkpoints.py", "scripts/analyze_b2_to_safebg_results.py",
        "scripts/compare_b2_to_safebg_mechanisms.py",
    }
    hashes = receipt.get("file_sha256", {})
    if not required <= hashes.keys():
        raise RuntimeError("Admission source identity is incomplete")
    for rel, digest in hashes.items():
        if sha256(inside(ROOT / rel)) != digest:
            raise RuntimeError(f"Admitted source changed: {rel}")
    return receipt


def wait_gpu():
    while subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True).strip():
        time.sleep(30)


def run_stage(name, command, status_path, status, env, timeout=None):
    status.update(state="running", stage=name, updated_at_unix=time.time())
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    with (ROOT / f"{status_path.stem}.{name}.console.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124
    (ROOT / f"{status_path.stem}.{name}.exitcode").write_text(str(code) + "\n", encoding="utf-8")
    if code != 0:
        raise RuntimeError(f"{name} failed with exit code {code}; artifacts retained")


def validate_coordinates(config, sanity_gate):
    from ue_framework.config import load_config
    from ue_framework.methods.semantic_coordinates import semantic_identity
    cfg = load_config(str(config))
    art = Path(cfg["methods"]["legacy_kproto_ret"]["semantic"]["artifact_dir"])
    identity = semantic_identity(cfg, ROOT)
    gate = json.loads(sanity_gate.read_text(encoding="utf-8"))
    if gate.get("state") != "passed" or gate.get("config_sha256") != sha256(config):
        raise RuntimeError("Five-step sanity receipt is absent or belongs to another config")
    if not (art / "strict_tsfl_channels.pt").is_file() or not (art / "safe_bg_selected_subspaces.pt").is_file():
        raise RuntimeError("Frozen Strict TSFL/Safe-BG artifacts are incomplete")
    summary = json.loads((art / "strict_tsfl_summary.json").read_text(encoding="utf-8"))
    if summary["final_selected_channels"] < 4:
        raise RuntimeError("Strict TSFL selected fewer than four channels")
    if summary["mean_S_t_selected"] <= summary["mean_S_t_unselected"]:
        raise RuntimeError("Strict TSFL target sensitivity gate failed")
    if summary["mean_S_nt_worst_selected"] >= summary["mean_S_nt_worst_unselected"]:
        raise RuntimeError("Strict TSFL worst-NT sensitivity gate failed")
    return identity, summary


def verify_noise(config, noise, expected_identity):
    from ue_framework.config import load_config
    from ue_framework.methods.semantic_coordinates import semantic_identity
    cfg = load_config(str(config))
    status = json.loads((noise / "status.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((noise / "diagnostics.json").read_text(encoding="utf-8"))
    method = cfg["methods"]["legacy_kproto_ret"]
    epochs, batch = int(method["universal_epochs"]), int(method["universal_batch_size"])
    expected = math.ceil(6095 / batch) * epochs
    if (status.get("state") != "complete" or status.get("smoke")
            or status.get("steps") != expected or status.get("epochs_completed") != epochs):
        raise RuntimeError("Formal B2-TO-Safe noise optimization is incomplete")
    if diagnostics.get("optimization_support_sources") != {"bbox_half_overlap": 6095 * epochs}:
        raise RuntimeError("Noise support coverage differs from formal B2-TO")
    if status.get("config_sha256") != sha256(config):
        raise RuntimeError("Noise config hash mismatch")
    if status.get("semantic_identity") != expected_identity or semantic_identity(cfg, ROOT) != expected_identity:
        raise RuntimeError("Noise semantic coordinates/source identity changed")
    for model in ("frozen", "adaptive"):
        if status.get(f"{model}_state_sha256_before") != status.get(f"{model}_state_sha256_after"):
            raise RuntimeError(f"{model} base changed during noise optimization")
    with (noise / "diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected:
        raise RuntimeError("Noise diagnostics step count mismatch")
    for row in rows:
        for key, value in row.items():
            if key != "selected_proto_ids" and not math.isfinite(float(value)):
                raise RuntimeError(f"Nonfinite noise diagnostic: {key}")
    retentions = [float(row["safe_direction_retention"]) for row in rows]
    median_retention = statistics.median(retentions)
    if median_retention < 0.10:
        raise RuntimeError(f"Full-run Safe-BG median retention below 0.10: {median_retention}")
    return {"steps": expected, "epochs": epochs, "median_of_step_median_retention": median_retention,
            "collapsed_ratio_mean": statistics.mean(float(r["collapsed_safe_direction_ratio"]) for r in rows),
            "semantic_identity": expected_identity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sanity-gate", required=True)
    parser.add_argument("--admission")
    parser.add_argument("--noise", required=True)
    parser.add_argument("--victim", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--clean-metrics")
    parser.add_argument("--formal-b2-to-metrics")
    parser.add_argument("--formal-b2-to-noise-csv")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, gate, noise, victim, status_path = map(inside,
        (args.config, args.sanity_gate, args.noise, args.victim, args.status))
    if noise == victim:
        raise ValueError("Noise and victim output roots must be distinct")
    from ue_framework.config import load_config
    cfg = load_config(str(config))
    if cfg["experiment"]["seeds"][0] != 0:
        raise ValueError("B2-TO-Safe is locked to seed 0")
    if args.check_only:
        if any(path.exists() for path in (noise, victim, status_path)):
            raise FileExistsError("A B2-TO-Safe output path already exists")
        method = cfg["methods"]["legacy_kproto_ret"]
        for output in (Path(cfg["platform"]["run_root"]),
                       Path(method["semantic"]["artifact_dir"])):
            inside(output)
        reference_read_path(method["semantic"]["safe_bg"]["reference_artifact_dir"])
        print("Safe-BG paths/config verified; no experiment started")
        return
    if args.smoke_only:
        if status_path.exists() or noise.exists() or victim.exists():
            raise FileExistsError("Refusing to overwrite a previous Safe-BG sanity result")
        admission = None
    else:
        if not args.admission:
            raise ValueError("Formal B2-TO-Safe requires the completed admission receipt")
        admission = inside(args.admission)
        receipt = validate_admission(config, admission)
        if any(path.exists() for path in (noise, victim, status_path)):
            raise FileExistsError("Refusing to overwrite any existing formal Safe-BG output")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {"state": "running", "stage": "initializing", "started_at_unix": time.time(),
              "noise": str(noise), "victim": str(victim), "config_sha256": sha256(config),
              "cleanup_policy": "retain_all", "smoke_only": args.smoke_only}
    with status_path.open("x", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT),
               YOLO_CONFIG_DIR=str(ROOT / ".ultralytics"), MPLCONFIGDIR=str(ROOT / ".mpl"),
               XDG_CACHE_HOME=str(ROOT / ".cache"), TORCH_HOME=str(ROOT / ".torch"),
               WANDB_MODE="disabled")
    try:
        if args.smoke_only:
            wait_gpu()
            run_stage("coordinates", [sys.executable, "scripts/build_b2_to_safe_coordinates.py",
                "--config", str(config), "--reference-artifact-dir",
                cfg["methods"]["legacy_kproto_ret"]["semantic"]["safe_bg"]["reference_artifact_dir"],
                "--output", cfg["methods"]["legacy_kproto_ret"]["semantic"]["artifact_dir"],
                "--seed", "0"], status_path, status, env)
            # This candidate config is copied and bounded only in train_noise_once's in-memory smoke config.
            run_stage("sanity", [sys.executable, "scripts/train_noise_once.py", "--config", str(config),
                "--seed", "0", "--smoke-max-images", "5", "--smoke-max-steps", "5",
                "--output", str(noise)], status_path, status, env, timeout=30 * 60)
            run_stage("sanity_gate", [sys.executable, "scripts/check_b2_to_safebg_sanity.py",
                "--config", str(config), "--noise", str(noise), "--output", str(status_path.parent)],
                status_path, status, env)
            status.update(state="complete", completed_at_unix=time.time(),
                          sanity_gate=str(status_path.parent / "sanity_gate.json"))
        else:
            if not (args.clean_metrics and args.formal_b2_to_metrics and args.formal_b2_to_noise_csv):
                raise ValueError("Final comparison requires clean, formal B2-TO, and B2-TO noise metrics")
            clean_metrics = reference_read_path(args.clean_metrics)
            formal_metrics = reference_read_path(args.formal_b2_to_metrics)
            formal_noise_csv = reference_read_path(args.formal_b2_to_noise_csv)
            identity, _ = validate_coordinates(config, gate)
            status["admitted_commit"] = receipt["candidate_commit"]
            wait_gpu()
            run_stage("noise", [sys.executable, "scripts/train_noise_once.py", "--config", str(config),
                "--seed", "0", "--output", str(noise)], status_path, status, env, timeout=6 * 3600)
            noise_receipt = verify_noise(config, noise, identity)
            status["noise_audit"] = noise_receipt
            run_stage("materialization", [sys.executable, "scripts/materialize_noise_once.py",
                "--config", str(config), "--seed", "0", "--noise", str(noise)], status_path, status, env)
            audit_path = status_path.parent / "materialization_audit.json"
            run_stage("pixel_audit", [sys.executable, "scripts/audit_b2_to_safebg_materialization.py",
                "--config", str(config), "--noise", str(noise), "--output", str(status_path.parent)],
                status_path, status, env, timeout=2 * 3600)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("state") != "passed":
                raise RuntimeError("Strict decoded-pixel and label audit failed")
            wait_gpu()
            run_stage("victim", [sys.executable, "scripts/run_isolated_victim.py", "--config",
                str(config), "--seed", "0", "--noise", str(noise), "--output", str(victim)],
                status_path, status, env)
            run_stage("checkpoint_evaluation", [sys.executable,
                "scripts/evaluate_b2_to_safebg_checkpoints.py", "--config", str(config),
                "--victim", str(victim), "--output", str(status_path.parent)], status_path, status, env)
            run_stage("results_analysis", [sys.executable, "scripts/analyze_b2_to_safebg_results.py",
                "--clean", str(clean_metrics), "--formal-b2-to", str(formal_metrics),
                "--safe-last", str(status_path.parent / "primary_last_metrics.json"),
                "--safe-best", str(status_path.parent / "secondary_best_metrics.json"),
                "--output", str(status_path.parent / "ablation_metrics.json")], status_path, status, env)
            run_stage("mechanism_comparison", [sys.executable, "scripts/compare_b2_to_safebg_mechanisms.py",
                "--formal-b2-to-csv", str(formal_noise_csv), "--safe-csv", str(noise / "diagnostics.csv"),
                "--output", str(status_path.parent / "mechanism_comparison.json")], status_path, status, env)
            status.update(state="complete", completed_at_unix=time.time())
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), completed_at_unix=time.time())
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
