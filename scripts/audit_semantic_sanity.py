"""Read-only checks of actual production sanity evidence, not scientific result metrics."""
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ue_framework.config import load_config
from ue_framework.data_utils import read_yolo_annotations
from ue_framework.methods.legacy_kproto_ret import _rectangle


ROOT = Path(__file__).resolve().parent.parent


def load(path):
    return json.loads(Path(path).read_text())


def main():
    evidence = {}; checks = {}
    for method in ('b2_original','b2','b2_t','b2_to'):
        run = ROOT/f'noise_sanity_{method}_5steps_v2_20260926'
        status = load(run/'status.json')
        rows = list(csv.DictReader((run/'diagnostics.csv').open()))
        finite = all(math.isfinite(float(v)) for row in rows for k,v in row.items() if k != 'selected_proto_ids')
        checks[method+'_runtime'] = (status['state']=='complete' and status['steps']==5 and finite and
            status['frozen_state_sha256_before']==status['frozen_state_sha256_after'] and
            status['adaptive_state_sha256_before']==status['adaptive_state_sha256_after'])
        evidence[method] = {'status':status, 'rows':rows}
        if method != 'b2_original':
            evidence[method]['fixed_pair'] = load(run/'fixed_pair.json')
    ref, candidate = evidence['b2_original']['rows'], evidence['b2']['rows']
    differences = [abs(float(a[k])-float(b[k])) for a,b in zip(ref,candidate)
                   for k in a if k != 'selected_proto_ids']
    max_diff = max(differences)
    a = torch.load(ROOT/'noise_sanity_b2_original_5steps_v2_20260926/global_params.pt',weights_only=True)
    b = torch.load(ROOT/'noise_sanity_b2_5steps_v2_20260926/global_params.pt',weights_only=True)
    coeff_diff = float((a['fourier_coeff']-b['fourier_coeff']).abs().max())
    checks['disabled_numerical_equivalence'] = max_diff <= 1e-6 and coeff_diff <= 1e-7
    a = torch.load(ROOT/'artifacts_v2/target_specific_channels.pt',weights_only=True)
    b = torch.load(ROOT/'artifacts_v2_repeat/target_specific_channels.pt',weights_only=True)
    checks['independent_full_build_selection'] = torch.equal(a['indices'],b['indices'])
    sensitivity_diff = max(float((a[k]-b[k]).abs().max()) for k in ('S_target','S_nt','specificity'))
    subs = load(ROOT/'artifacts_v2/subspace_summary.json')
    checks['all_classes_90pct_variance'] = all(v['explained_variance']>=.90 for v in subs.values())
    checks['positive_term_gradients'] = all(float(row[k])>0 for m in ('b2_t','b2_to')
        for row in evidence[m]['rows'] for k in ('grad_BG_sel_norm','grad_rel_sel_norm','grad_orth_raw_norm','grad_adapt_norm'))
    checks['TO_weighted_orth_gradient'] = all(float(row['grad_orth_norm'])>0 for row in evidence['b2_to']['rows'])
    config = load_config(str(ROOT/'ue_framework/configs/exp_voc_person_b2_t.yaml'))
    data = Path(config['data']['dataset_root'])
    source_dir = data/config['data']['train_images']
    source_by_stem = {p.stem:p for p in source_dir.iterdir() if p.is_file()}
    pixels = {}
    for method in ('b2_t','b2_to'):
        counts = {'target_pixels':0, 'unchanged_target':0, 'outside_changed':0,
                  'overlap_max':0, 'global_max':0}
        for preview in (ROOT/f'noise_sanity_{method}_5steps_v2_20260926/preview').glob('*.png'):
            clean = cv2.imread(str(source_by_stem[preview.stem])); poison = cv2.imread(str(preview))
            h,w = clean.shape[:2]; target = np.zeros((h,w),bool); other = target.copy()
            for ann in read_yolo_annotations(str(data/config['data']['train_labels']/(preview.stem+'.txt'))):
                box = _rectangle((h,w),ann['bbox'])[0,0].numpy().astype(bool)
                if int(ann['cls'])==14:target |= box
                else:other |= box
            delta = np.abs(clean.astype(np.int16)-poison.astype(np.int16)).max(2)
            counts['target_pixels'] += int(target.sum())
            counts['unchanged_target'] += int((delta[target]==0).sum())
            counts['outside_changed'] += int((delta[~target]!=0).sum())
            counts['overlap_max'] = max(counts['overlap_max'],int(delta[target & other].max(initial=0)))
            counts['global_max'] = max(counts['global_max'],int(delta.max()))
        pixels[method] = counts
        checks[method+'_png_support_budget'] = (counts['target_pixels']>0 and counts['unchanged_target']==0
            and counts['outside_changed']==0 and counts['overlap_max']<=8 and counts['global_max']<=16)
    pair_t = evidence['b2_t']['fixed_pair']; pair_to = evidence['b2_to']['fixed_pair']
    checks['T_BG_rel_decrease_fixed_input'] = all(pair_t['after_5'][k]<pair_t['initial'][k] for k in ('L_BG_sel','L_rel_sel'))
    checks['T_non_target_drift_not_above_B2'] = pair_t['after_5']['L_ret'] <= evidence['b2']['fixed_pair']['after_5']['L_ret']
    checks['TO_orth_decrease_fixed_input'] = pair_to['after_5']['L_orth_worst']<pair_to['initial']['L_orth_worst']
    checks['TO_displacement_not_collapsed'] = pair_to['after_5']['target_displacement_norm']>1e-3
    checks['TO_gradient_not_dominant'] = all(float(row['grad_orth_norm']) <= float(row['grad_attack_norm'])
        for row in evidence['b2_to']['rows'])
    result = {'checks':checks, 'all_passed':all(checks.values()), 'baseline_diagnostics_max_difference':max_diff,
        'baseline_coeff_max_difference':coeff_diff, 'independent_build_sensitivity_max_difference':sensitivity_diff,
        'preview_pixel_audit':pixels, 'subspaces':subs, 'evidence':evidence}
    (ROOT/'semantic_sanity_audit_v2.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'},indent=2))


if __name__=='__main__':
    main()
