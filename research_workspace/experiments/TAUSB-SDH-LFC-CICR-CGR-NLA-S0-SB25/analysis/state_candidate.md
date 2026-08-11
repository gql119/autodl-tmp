# STATE candidate: TAUSB-SDH-LFC-CICR-CGR-NLA-S0-SB25

Proposed disposition: record as a **failed hiding-only path**, without changing Current Best.

- `HIDING-S0-SB25-R1` completed at code commit `d244c3270eb24d7a6515e79ff643cb015ebb0bb9`.
- `hf_subband_scale=0.25` reduced held-out high-frequency energy median from `0.671792` to
  `0.034223`, passing the spectral cap.
- It failed secret retrieval (`0.424479 < 0.90`) and primary relative L1 margin
  (`0.067159 < 0.20`).
- RMS CV remains descriptive and did not determine the failure.
- Mechanism, victim, AP50, unlearnability, robustness, transfer and SOTA claims are not supported.
- Candidate next test: a separately approved, otherwise matched hiding-only run at
  `hf_subband_scale=0.50`.

Disposition approved by the user on 2026-08-11 and applied to `research_workspace/STATE.md`.
Current Best remains unchanged.
