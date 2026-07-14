# C1 Real-YOLO Adapter Implementation

## Adapter

`oa_lgc/yolo_adapter.py` introduces `YOLOFunctionalAdapter`. Its backend is explicitly `real_ultralytics_yolo`; no proxy object or fallback branch is present.

- Functional backend: `torch.func.functional_call`.
- Inner loss: native `DetectionModel.loss`, returning box, classification, and DFL components.
- Inner optimizer: SGD, learning rate `1e-4`, momentum 0, weight decay 0.
- Mixed derivative: `torch.autograd.grad(..., create_graph=True)` for the required Mode A trajectory.
- Buffer mode: cloned. Every trajectory receives independent copies; query evaluation clones trajectory buffers again.
- Base model: frozen and hashed before/after. Functional updates never write back.

## Exact parameter selection

The implementation locates modules by registered object identity and expands their recursive parameters. It does not silently select by fuzzy substrings.

| Mode | Exact modules | Tensors | Parameters | Hash prefix |
| --- | --- | ---: | ---: | --- |
| classification_head_only | Detect `cv3` | 24 | 373,308 | `5edc34f7` |
| detection_head | Detect excluding fixed DFL integral | 48 | 755,196 | `2eb9f78d` |
| selected_neck_and_head | layers 15, 18, 21, 22 | 84 | 1,409,148 | `bd119ec4` |
| full_model | all eligible parameters | 183 | 3,014,732 | `369244ac` |

`model.22.dfl.conv.weight` is the fixed 16-bin integral vector. It is manifested but omitted from trainable fast sets.

## Assignment and query loss

The adapter recreates the native criterion's TAL input preparation and calls its real assigner. For the pilot path, `target_labels`, `target_scores`, `fg_mask`, `target_gt_idx`, boxes, anchors, and strides are detached once from the clean query/base model and reused.

The classwise query loss uses only units with positive reference target score for that class. It sums BCE-with-logits over those units and divides by target-score mass. Classes without reference positives are marked invalid and are not zero-filled into averages.

TAL top-k/matching/index decisions are a declared stop-gradient boundary. Prediction scores and losses outside the discrete assignment retain gradients.

## Protect-only path

The C1 diagnostic uses clean query pixels for initial, clean-fast, and poison-fast losses. Therefore its only delta path is:

`clean query loss -> poison fast parameters -> native poison-support inner loss -> object-aligned carrier -> delta_obj`

Carrier, authorization, and regularizer weights are separately excluded when measuring `protect_only_grad_norm`.
