"""Resume admitted downstream stages after the parent-only import-path failure.

The original supervisor, admission receipt, successful noise, and scientific code
remain unchanged. This entry binds the project path in the supervisor itself.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run_b2_to_auto_pipeline import inside, sha256, validate_admission, verify_noise, wait_gpu, run_stage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'admission', 'noise', 'victim', 'status', 'failed-status'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    config, admission, noise, victim, status_path, failed_path = map(inside,
        (args.config, args.admission, args.noise, args.victim, args.status, args.failed_status))
    if args.seed != 0 or noise == victim:
        raise ValueError('Admitted seed0 and distinct outputs required')
    receipt = validate_admission(config, admission)
    failed = json.loads(failed_path.read_text())
    if failed.get('state') != 'failed' or failed.get('stage') != 'noise' or failed.get('error') != "No module named 'ue_framework'":
        raise RuntimeError('This continuation is limited to the diagnosed parent import failure')
    if failed.get('noise') != str(noise) or failed.get('victim') != str(victim) or failed.get('config_sha256') != sha256(config):
        raise RuntimeError('Failed pipeline identity differs')
    exit_path = ROOT / (failed_path.stem + '.noise.exitcode')
    if exit_path.read_text().strip() != '0':
        raise RuntimeError('Cannot reuse an unsuccessful noise optimization')
    noise_status = json.loads((noise / 'status.json').read_text())
    if noise_status.get('seed') != 0 or noise_status.get('duration_seconds', float('inf')) > 6 * 3600:
        raise RuntimeError('Noise seed or six-hour budget differs')
    verify_noise(config, noise)
    if victim.exists() or status_path.exists():
        raise FileExistsError('Continuation/victim output already exists')
    if args.check_only:
        print(json.dumps({'noise_verification': 'passed', 'steps': noise_status['steps'],
                          'epochs': noise_status['epochs_completed'],
                          'duration_hours': noise_status['duration_seconds'] / 3600,
                          'config_sha256': sha256(config),
                          'candidate_commit': receipt['candidate_commit'],
                          'experiment_started': False}))
        return
    status = {'state': 'running', 'stage': 'verified_existing_noise',
              'started_at_unix': time.time(), 'noise': str(noise), 'victim': str(victim),
              'config_sha256': sha256(config), 'admitted_commit': receipt['candidate_commit'],
              'failed_pipeline_preserved': str(failed_path),
              'continuation_code_sha256': sha256(Path(__file__)),
              'noise_retrained': False, 'cleanup_policy': 'retain_all'}
    with status_path.open('x') as f:
        json.dump(status, f, indent=2)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT),
               YOLO_CONFIG_DIR=str(ROOT/'.ultralytics'), MPLCONFIGDIR=str(ROOT/'.mpl'),
               XDG_CACHE_HOME=str(ROOT/'.cache'), TORCH_HOME=str(ROOT/'.torch'), WANDB_MODE='disabled')
    common = ['--config', str(config), '--seed', str(args.seed)]
    try:
        run_stage('materialization', [sys.executable, 'scripts/materialize_noise_once.py',
                  *common, '--noise', str(noise)], status_path, status, env)
        status.update(stage='waiting_gpu', updated_at_unix=time.time())
        status_path.write_text(json.dumps(status, indent=2))
        wait_gpu()
        # The production victim entry retains its complete _full_data gate.
        run_stage('victim', [sys.executable, 'scripts/run_isolated_victim.py',
                  *common, '--noise', str(noise), '--output', str(victim)], status_path, status, env)
        status.update(state='complete', completed_at_unix=time.time())
    except BaseException as exc:
        status.update(state='failed', error=str(exc), completed_at_unix=time.time())
        raise
    finally:
        status_path.write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
