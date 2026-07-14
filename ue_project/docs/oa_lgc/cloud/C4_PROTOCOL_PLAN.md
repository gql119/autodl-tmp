# C4 Protocol Plan

## Required protocol

- Training: VOC2007 trainval plus VOC2012 trainval.
- Test: VOC2007 test only.
- Train/test overlap: zero.
- OA-LGC support/query samples: training manifest only.
- Test data: final evaluation only; never delta optimization or hyperparameter selection.

The audit first searches local configurations, split manifests, raw VOC directories, checkpoint metadata, historical training curves, and evaluation records. Missing components must block the stage; no split guessing, test repurposing, or silent download is allowed.

If the data protocol is complete, a clean YOLOv8n baseline must provide per-epoch person AP50 and non-target mean AP50. `E_pilot` is the smallest candidate among 20/40/60/80/100 where both reach at least 80% of their final clean values.

If any prerequisite is absent, the stage produces a blocked audit artifact and does not start training or C5.
