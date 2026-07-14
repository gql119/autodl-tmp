# C4 Clean Baseline Report

## Formal baseline

Status: not run because the data protocol is blocked.

`E_pilot`: not selected.

## Historical evidence recovered

The repository contains a reproducible clean-from-scratch YOLOv8n-style baseline for a deterministic 800-train/200-validation subset of VOC2007 trainval:

- epochs/imgsz/batch: 50/416/16;
- seed: 0;
- pretrained/resume: false/false;
- person AP50: 0.308194;
- non-target mean AP50: 0.082198;
- all mAP50: 0.093498;
- worst non-target AP50: 0.000434;
- best checkpoint SHA256: `c5a731edd3ed75bca676e365bb5eb175ea404fb439bec3f9872b40999119ece7`, verified against the local file.

This checkpoint is not C4-protocol eligible. Its validation set is another VOC2007 trainval subset rather than VOC2007 test, VOC2012 is absent, and its per-epoch CSV records only aggregate mAP rather than person/non-target AP curves. Consequently it cannot determine the required 80%-convergence `E_pilot`.

The C4 artifact copies the historical curve and classwise AP with `protocol_eligible=0` and preserves the source protocol label. These records are context, not the pilot baseline.
