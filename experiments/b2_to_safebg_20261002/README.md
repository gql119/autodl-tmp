# B2-TO-Safe

Independent structural ablation of the completed formal B2-TO eps8_overlap4 run. The only scientific changes are (1) Strict TSFL channel selection and (2) removing the per-instance top-3 non-target semantic projection from the selected-space context background direction. The existing angular `L_orth` source and parameters are frozen at `lambda_orth=0.0005`, `orth_worst_k=3`, `overlap_orth_weight=2.0`, and `orth_eps=1e-12`.

`b2_to_safe_config_diff_before_run.json` records the exact formal B2-TO config hash (`597871ef…`) and the approved path/settings differences before any SafeBG run. `baseline_formal_b2_to_config.yaml` is the exact config snapshot used by the completed formal B2-TO eps8_overlap4 run. The new config keeps the same dataset, target, seed, full noise schedule (24 epochs/9144 steps), optimizer, losses/weights, adaptive learner and 200-epoch randomized victim protocol; it changes only output/artifact locations and the requested Strict TSFL/Safe-BG configuration.

Strict TSFL and selected-space per-class PCA bases are built from clean training data with the same frozen reference, GT evidence, feature layer and existing PCA rules. Final validation is never used to build or select channels. The formal full-space bases are reused byte-for-byte for `L_orth`; the new selected-space bases are an additional Safe-BG artifact.

The matched baseline uses the completed formal `eps8_overlap4` run (main epsilon 8/255, overlap 4/255). For this ablation, the original B2-TO channel indices and projected subspaces are separately frozen for `L_orth`; Strict TSFL channels are used only by BG/rel and Safe-BG. This keeps the orthogonality objective's coordinates unchanged while testing the requested structural edits.

Preflight outcome: blocked at Strict TSFL selection before the five-step perturbation smoke. The frozen training-split sensitivity scan reached the selector, where the preregistered 30% target gate and 50th-percentile worst-non-target safety gate left zero eligible channels. The requested minimum is four, so the experiment stopped without relaxing either gate. The coordinate-build status, console log, and failure receipt are retained. No noise optimization, materialization, or victim training was started.

Remote focused validation passed (2 unit tests; config-diff audit). This directory records an implementation and a failed admission gate, not B2-TO-Safe experimental results.
