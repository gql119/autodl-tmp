# Hiding gate decision: HIDING-S0-R2

Decision: **FAIL / block mechanism and victim stages**.

The run completed normally and passed recovery, L1 margin, pixel cosine, finite, Linf, support,
retrieval and non-target leakage checks. It failed the two independent carrier checks:

- channel RMS CV: `0.018599 / 0.010226 / 0.015434`, required every channel `>= 0.05`;
- high-frequency energy median: `0.671792`, required `<= 0.40`.

The evidence, hashes, full threshold table, interpretation and single next experiment are in
`research_workspace/experiments/TAUSB-SDH-LFC-CICR-CGR-NLA-S0/analysis/result-TAUSB-SDH-LFC-CICR-CGR-NLA-S0.md`.

No detector mechanism, victim training, AP50 evaluation, robustness evaluation, or UE claim is
permitted from this result.

