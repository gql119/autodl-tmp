"""Verify the completed official T noise and the full materialization before victim."""
import csv
import hashlib
import json
import math
from pathlib import Path
from ue_framework.config import load_config
from run_isolated_victim import _full_data

ROOT=Path(__file__).resolve().parent.parent
CONFIG=ROOT/'ue_framework/configs/exp_voc_person_b2_t.yaml'
NOISE=ROOT/'noise_full_b2_t_seed0_20260926'


def main():
    cfg=load_config(str(CONFIG))
    status=json.loads((NOISE/'status.json').read_text())
    diagnostics=json.loads((NOISE/'diagnostics.json').read_text())
    rows=list(csv.DictReader((NOISE/'diagnostics.csv').open()))
    if status['state']!='complete' or status['smoke'] or status['steps']!=9144 or status['epochs_completed']!=24:
        raise RuntimeError('Official T optimization incomplete')
    if diagnostics['optimization_support_sources']!={'bbox_half_overlap':146280}:
        raise RuntimeError('Noise support source count mismatch')
    if not all(math.isfinite(float(v)) for row in rows for k,v in row.items() if k!='selected_proto_ids'):
        raise RuntimeError('Nonfinite official noise diagnostics')
    for model in ('frozen','adaptive'):
        if status[f'{model}_state_sha256_before']!=status[f'{model}_state_sha256_after']:
            raise RuntimeError('Base state changed')
    stage,train_count,val_count=_full_data(cfg,0,NOISE,str(CONFIG))
    if (train_count,val_count)!=(16551,4952):
        raise RuntimeError('Unexpected dataset size')
    result={'state':'passed','stage':str(stage),'train_count':train_count,'clean_val_count':val_count,
            'noise_global_params_sha256':hashlib.sha256((NOISE/'global_params.pt').read_bytes()).hexdigest(),
            'semantic_identity':status['semantic_identity'],
            'checks':'All stems, labels bytes, noise/config/core/adaptive/semantic/artifact hashes; all target pixels changed; outside0; global16/255 overlap8/255; non-target-only unchanged; clean val byte-identical'}
    (ROOT/'full_data_preflight_b2_t_20260926.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
