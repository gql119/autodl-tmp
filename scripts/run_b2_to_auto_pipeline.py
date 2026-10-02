"""Admitted noise -> corresponding materialization -> victim, without manual relaunch."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
# Bind project imports in this supervisor, as well as in child processes.
sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Pipeline path outside isolated project: {path}')
    return path


def validate_admission(config, admission):
    receipt = json.loads(admission.read_text())
    if receipt.get('sanity_gate') != 'passed' or receipt.get('scientific_review') != 'scientifically_correct':
        raise RuntimeError('Current candidate has not passed sanity and scientific review')
    if receipt.get('config_sha256') != sha256(config):
        raise RuntimeError('Admission config identity differs')
    required = {'ue_framework/methods/b2_semantic.py',
                'ue_framework/methods/semantic_coordinates.py',
                'ue_framework/methods/orth_reference_design.py',
                'ue_framework/methods/b2_adaptive.py',
                'ue_framework/methods/adaptive_learner.py',
                'scripts/train_noise_once.py', 'scripts/materialize_noise_once.py',
                'scripts/run_isolated_victim.py'}
    hashes = receipt.get('file_sha256', {})
    if not required <= hashes.keys():
        raise RuntimeError('Admission source identity is incomplete')
    for rel, digest in hashes.items():
        if sha256(inside(ROOT / rel)) != digest:
            raise RuntimeError(f'Admitted source changed: {rel}')
    return receipt


def wait_gpu():
    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                   '--format=csv,noheader'], text=True).strip():
        time.sleep(30)


def run_stage(name, command, status_path, status, env, timeout=None):
    status.update(state='running', stage=name, updated_at_unix=time.time())
    status_path.write_text(json.dumps(status, indent=2))
    with (ROOT / f'{status_path.stem}.{name}.console.log').open('x') as log:
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
    (ROOT / f'{status_path.stem}.{name}.exitcode').write_text(str(code) + '\n')
    if code != 0:
        raise RuntimeError(f'{name} failed with exit code {code}; artifacts retained')


def verify_noise(config, noise):
    from ue_framework.config import load_config
    cfg = load_config(str(config))
    status = json.loads((noise / 'status.json').read_text())
    diagnostics = json.loads((noise / 'diagnostics.json').read_text())
    method = cfg['methods']['legacy_kproto_ret']
    epochs, batch = int(method['universal_epochs']), int(method['universal_batch_size'])
    expected = math.ceil(6095 / batch) * epochs
    if status['state'] != 'complete' or status['smoke'] or status['steps'] != expected or status['epochs_completed'] != epochs:
        raise RuntimeError('Official noise optimization incomplete')
    if diagnostics['optimization_support_sources'] != {'bbox_half_overlap': 6095 * epochs}:
        raise RuntimeError('Noise support source coverage mismatch')
    if status['config_sha256'] != sha256(config):
        raise RuntimeError('Noise config changed')
    with (noise / 'diagnostics.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != expected:
        raise RuntimeError('Noise diagnostics step count mismatch')
    for row in rows:
        for key, value in row.items():
            if key != 'selected_proto_ids' and not math.isfinite(float(value)):
                raise RuntimeError(f'Nonfinite noise diagnostic: {key}')
    for model in ('frozen', 'adaptive'):
        if status[f'{model}_state_sha256_before'] != status[f'{model}_state_sha256_after']:
            raise RuntimeError('Noise optimization changed a base model')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--noise', required=True)
    parser.add_argument('--victim', required=True)
    parser.add_argument('--status', required=True)
    parser.add_argument('--budget-reference-poison-root',
                        help='Optional matched B2-TO materialized train root for paired pixel statistics')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config, admission, noise, victim, status_path = map(inside,
        (args.config, args.admission, args.noise, args.victim, args.status))
    if args.seed != 0 or noise == victim:
        raise ValueError('This admitted experiment requires seed0 and distinct outputs')
    receipt = validate_admission(config, admission)
    if any(path.exists() for path in (noise, victim, status_path)):
        raise FileExistsError('Pipeline or experiment output already exists')
    if args.check_only:
        print('Admission/source identities and fresh outputs verified; no experiment started')
        return
    status = {'state': 'running', 'stage': 'waiting_gpu', 'started_at_unix': time.time(),
              'noise': str(noise), 'victim': str(victim), 'config_sha256': sha256(config),
              'admitted_commit': receipt['candidate_commit'], 'cleanup_policy': 'retain_all'}
    # Exclusive creation prevents two copies from starting the same pipeline.
    with status_path.open('x') as f:
        json.dump(status, f, indent=2)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT),
               YOLO_CONFIG_DIR=str(ROOT/'.ultralytics'), MPLCONFIGDIR=str(ROOT/'.mpl'),
               XDG_CACHE_HOME=str(ROOT/'.cache'), TORCH_HOME=str(ROOT/'.torch'), WANDB_MODE='disabled')
    common = ['--config', str(config), '--seed', str(args.seed)]
    try:
        wait_gpu()
        run_stage('noise', [sys.executable, 'scripts/train_noise_once.py', *common,
                           '--output', str(noise)], status_path, status, env, timeout=6*3600)
        verify_noise(config, noise)
        run_stage('materialization', [sys.executable, 'scripts/materialize_noise_once.py',
                                     *common, '--noise', str(noise)], status_path, status, env)
        wait_gpu()
        # run_isolated_victim performs the production full-data preflight itself.
        # It then trains the victim on this exact materialized config/noise identity.
        run_stage('victim', [sys.executable, 'scripts/run_isolated_victim.py', *common,
                            '--noise', str(noise), '--output', str(victim),
                            *(['--budget-reference-poison-root', args.budget_reference_poison_root]
                              if args.budget_reference_poison_root else [])],
                  status_path, status, env)
        status.update(state='complete', completed_at_unix=time.time())
    except BaseException as exc:
        status.update(state='failed', error=str(exc), completed_at_unix=time.time())
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
