# C1 Real-YOLO Adapter Report

## Result

Status: `pass`.

Authoritative artifact: `artifacts/oa_lgc/cloud/20260714_143121_C1_0/`.

| Gate | Result | Evidence |
| --- | --- | --- |
| Real YOLO forward | pass | Training output has native boxes/scores/FPN features |
| Native detection loss | pass | Positive finite box, classification, and DFL losses |
| classification_head_only J=1 | pass | 24 fast tensors; native full detection inner loss |
| detection_head J=1 | pass | 48 fast tensors; native full detection inner loss |
| selected_neck_and_head runnable | pass | 84 fast tensors; one J=1 run completed |
| full_model interface | pass | 183 eligible tensors manifested; no required update run |
| Base state hash | pass | before=after=`25c0ad56...eec28c` |
| Independent clean/poison states | pass | no selected parameter or buffer storage alias |
| Protect-only mixed gradient | pass | norm `12.078218460083008`, finite, no poison query pixels |
| No proxy fallback | pass | backend `real_ultralytics_yolo` |
| Reproducibility | pass | max selected-parameter difference `0.0`, tolerance `1e-7` |
| NaN/Inf | pass | none observed |

The fixed reference episode used support ID `000009` and query ID `000021`; overlap is zero. It contained 26 target reference-positive units, target score mass 12.885307, and 33 total foreground units.

Gradient decomposition:

- protect-only: 12.078218
- carrier-only: 9.969158
- authorization-only: 0.0
- regularizer-only: 0.000035718
- total: 15.904286

No non-target class met the C1 episode's support/query count requirement, so the authorization component was correctly invalid/zero for this one engineering episode. C1 does not use this as evidence about non-target behavior; C2 must sample episodes that contain at least one valid non-target class.

Peak allocated GPU memory was 284,306,432 bytes and total smoke time was 5.40 seconds. The complete regression command passed: `107 passed in 9.71s`.

No method-effectiveness claim follows from C1. It establishes only that the real-detector functional engineering path is valid.
