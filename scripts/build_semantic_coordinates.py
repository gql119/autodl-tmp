"""Build TSFL and class PCA once using only clean TRAIN and the frozen B2 reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from ultralytics import YOLO

from ue_framework.config import load_config
from ue_framework.data_utils import list_images, label_path_for_image, load_image_rgb_float, read_yolo_annotations
from ue_framework.methods.legacy_kproto_ret import LegacyKProtoTrainer, _rectangle, _feature_mask
from ue_framework.methods.semantic_coordinates import select_channels, class_subspace, gt_evidence, sha256, CLASSES


def state_hash(model):
    d = hashlib.sha256()
    for k, v in model.state_dict().items():
        d.update(k.encode()); d.update(v.detach().cpu().numpy().tobytes())
    return d.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    output = Path(args.output).resolve()
    root = Path(__file__).resolve().parent.parent
    if not output.is_relative_to(root):
        raise ValueError('Output outside isolated project')
    output.mkdir(exist_ok=False)
    cfg = load_config(args.config)
    method = cfg['methods']['legacy_kproto_ret']; s = method['semantic']
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    model = YOLO(cfg['surrogate']['ckpt']).model.cuda().eval()
    trainer = LegacyKProtoTrainer(cfg, method, torch.device('cuda'), model)
    before = state_hash(model)
    data = Path(cfg['data']['dataset_root'])
    labels = data/cfg['data']['train_labels']
    images = sorted(list_images(str(data/cfg['data']['train_images'])))
    counts = {'target_objects': 0, 'non_target_objects': 0, 'images': len(images)}
    pooled = {c: [] for c in range(20) if c != trainer.target_class_id}
    nt_gt_counts = {c: 0 for c in pooled}
    st = sn = None
    manifest = hashlib.sha256(); started = time.time()
    write(output/'status.json', {'state': 'running', 'images_total': len(images), 'started': started})
    for i, path in enumerate(images):
        image = load_image_rgb_float(path)
        label = label_path_for_image(path, str(labels))
        annotations = read_yolo_annotations(label)
        manifest.update((Path(path).stem+':'+sha256(path)+':'+sha256(label)+'\n').encode())
        if annotations:
            tensor = trainer._to_tensor(image).requires_grad_(True)
            trainer.multi_features.clear()
            decoded = trainer._forward_raw(tensor)
            feature = trainer.multi_features[trainer.feature_layer]
            if st is None:
                st = torch.zeros(feature.shape[1], dtype=torch.float64)
                sn = torch.zeros_like(st)
            for ann in annotations:
                c = int(ann['cls'])
                if c in pooled:
                    nt_gt_counts[c] += 1
                    mask = _feature_mask(_rectangle(image.shape[:2], ann['bbox']), feature, image.shape[:2], trainer.imgsz)
                    values = trainer._pooled(feature.detach(), [mask])
                    if values:
                        pooled[c].append(values[0].cpu())
            evidences = gt_evidence(decoded, annotations, image.shape[:2], trainer.imgsz)
            for c, evidence in evidences:
                gradient = torch.autograd.grad(evidence, feature, retain_graph=True)[0]
                sensitivity = gradient.detach().abs().mean(dim=(0, 2, 3)).double().cpu()
                if not torch.isfinite(sensitivity).all():
                    raise RuntimeError('Nonfinite channel sensitivity')
                if c == trainer.target_class_id:
                    st += sensitivity; counts['target_objects'] += 1
                else:
                    sn += sensitivity; counts['non_target_objects'] += 1
            del feature, decoded, tensor, evidences, gradient
        if (i+1) % 100 == 0:
            write(output/'status.json', {'state': 'running', 'images_done': i+1,
                  'images_total': len(images), 'elapsed_seconds': time.time()-started, **counts})
            print(f'{i+1}/{len(images)} elapsed={time.time()-started:.1f}', flush=True)
    if state_hash(model) != before:
        raise RuntimeError('Frozen state changed during coordinate build')
    st /= counts['target_objects']; sn /= counts['non_target_objects']
    indices, sth, snh, score = select_channels(st, sn, s['target_channel_fraction'], s['lambda_specificity'])
    repeat = select_channels(st.clone(), sn.clone(), s['target_channel_fraction'], s['lambda_specificity'])[0]
    if not torch.equal(indices, repeat):
        raise RuntimeError('Nondeterministic channel selection')
    identity = {'feature_layer': trainer.feature_layer, 'channels': len(st), 'seed': args.seed,
        'checkpoint_sha256': sha256(cfg['surrogate']['ckpt']), 'frozen_state_sha256': before,
        'clean_train_manifest_sha256': manifest.hexdigest(), 'images': len(images),
        'data_root': str(data), 'train_images': cfg['data']['train_images'],
        'train_labels': cfg['data']['train_labels'], 'validation_used': False,
        'builder_sha256': sha256(__file__),
        'coordinate_code_sha256': sha256(root/'ue_framework/methods/semantic_coordinates.py'),
        'settings': {k:s[k] for k in ('target_channel_fraction','lambda_specificity',
                                     'subspace_variance','subspace_min_rank','subspace_max_rank_fraction')}}
    tsfl = {'indices': indices, 'S_target': st, 'S_nt': sn, 'S_target_normalized': sth,
            'S_nt_normalized': snh, 'specificity': score, 'identity': identity}
    torch.save(tsfl, output/'target_specific_channels.pt')
    subspaces = {c: class_subspace(pooled[c], s['subspace_variance'], s['subspace_min_rank'],
                                  s['subspace_max_rank_fraction']) for c in pooled}
    torch.save({'classes':subspaces, 'identity':identity}, output/'non_target_subspaces.pt')
    chosen = torch.zeros(len(st), dtype=torch.bool); chosen[indices] = True
    summary = {'num_channels_total': len(st), 'num_channels_selected': len(indices),
        'target_channel_fraction': s['target_channel_fraction'], 'indices': indices.tolist(),
        'mean_target_sensitivity_selected': float(st[chosen].mean()),
        'mean_non_target_sensitivity_selected': float(sn[chosen].mean()),
        'mean_target_sensitivity_unselected': float(st[~chosen].mean()),
        'mean_non_target_sensitivity_unselected': float(sn[~chosen].mean()),
        'mean_S_target_selected': float(sth[chosen].mean()), 'mean_S_nt_selected': float(snh[chosen].mean()),
        'mean_S_target_unselected': float(sth[~chosen].mean()), 'mean_S_nt_unselected': float(snh[~chosen].mean()),
        'specificity_selected_mean': float(score[chosen].mean()),
        'channel_selection_overlap_with_previous_run': 1.,
        'determinism_test': 'two stable selections from the same full-training sensitivity tensors; not two independent calibrations',
        'aggregation': 'per GT mean spatial absolute gradient, then equal GT average', **counts}
    write(output/'tsfl_summary.json', summary)
    write(output/'subspace_summary.json', {CLASSES[c]: {k:v for k,v in x.items() if k not in ('basis','mean')} |
          {'gt_count':nt_gt_counts[c], 'invalid_pool_count':nt_gt_counts[c]-x['count']} for c,x in subspaces.items()})
    write(output/'coordinates_identity.json', identity)
    write(output/'status.json', {'state':'complete', 'images_done':len(images),
          'duration_seconds':time.time()-started, 'frozen_hash_unchanged':True, **counts})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
