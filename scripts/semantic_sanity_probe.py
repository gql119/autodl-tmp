"""Fixed-input paired diagnostics after 5 production outer steps; never victim metrics."""
import json
from pathlib import Path
import random

import numpy as np
import torch
from ue_framework.data_utils import load_image_rgb_float, label_path_for_image, read_yolo_annotations


def fixed_pair(trainer, image_dir, label_dir, initial_coeff, initial_suppress, params_path, seed):
    images = trainer._collect_target_images(image_dir, label_dir)[:16]
    items = []
    for path in images:
        im = load_image_rgb_float(path)
        ann = read_yolo_annotations(label_path_for_image(path,label_dir))
        support, _, _ = trainer._build_support(im.shape, ann, image_path=path)
        items.append((im,ann,support))
    saved = torch.load(params_path, map_location='cpu', weights_only=True)
    prototypes = torch.load(saved['prototype_cache'], map_location=trainer.device, weights_only=True)['prototypes']
    trained = trainer.fourier_coeff.detach().clone()
    trained_suppress = trainer.suppress_small.detach().clone()
    old_adaptive_calls = getattr(trainer,'_adaptive_calls',0)
    old_semantic_calls = getattr(trainer,'_semantic_calls',0)
    old_epoch = trainer._epoch
    results = {}
    trainer._epoch = 0
    # Identical last-active coordinates, RNG, adaptive image, and EOT for both sides.
    try:
        for label,coeff,suppress in [('initial',initial_coeff,initial_suppress),('after_5',trained,trained_suppress)]:
            with torch.no_grad():
                trainer.fourier_coeff.copy_(coeff); trainer.suppress_small.copy_(suppress)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            trainer._adaptive_calls = 0; trainer._semantic_calls = 0
            result = trainer._batch_loss(items,prototypes)
            if result is None:
                raise RuntimeError('Fixed probe had no target features')
            results[label] = {k:float(v.detach()) for k,v in result[1].items() if torch.is_tensor(v)}
            del result
    finally:
        with torch.no_grad():
            trainer.fourier_coeff.copy_(trained); trainer.suppress_small.copy_(trained_suppress)
        trainer._adaptive_calls = old_adaptive_calls; trainer._semantic_calls = old_semantic_calls
        trainer._epoch = old_epoch
    results.update({'images':images, 'sample_count':len(items), 'seed':seed,
                    'comparison':'same last-active coordinates, epoch0, image0 adaptive query, one fixed EOT'})
    Path(params_path).with_name('fixed_pair.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
