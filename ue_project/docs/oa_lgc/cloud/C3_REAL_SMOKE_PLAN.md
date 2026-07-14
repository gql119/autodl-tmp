# C3 Real-YOLO End-to-End Smoke Plan

## Matrix

| Run | J | Fast mode | Outer steps | Episodes | create_graph |
| --- | ---: | --- | ---: | ---: | --- |
| A | 1 | classification_head_only | 3 | 3 | true |
| B | 1 | detection_head | 3 | 3 | true |
| C | 3 | classification_head_only | 2 | 2 | true |
| D | 3 | detection_head | 2 | 2 | true |
| E | 5 | classification_head_only | 1 | 1 | true |

Each outer step uses one disjoint support/query pair with a shared non-target class. Every run starts from the same seed-controlled `delta_obj` and uses the same carrier, native detection loss, fixed clean-query reference assignment, gain definitions, and Adam outer learning rate `0.001`.

Run A is repeated with the same seed. IDs and reference-positive counts must be identical, gain differences must be at most `1e-6`, and final delta difference at most `1e-7`.

J=5 detection-head is optional and is not part of the pass gate when J=5 classification-head succeeds. No RCDS, QP, ALCE context, feature collision, or proxy fallback is permitted.
