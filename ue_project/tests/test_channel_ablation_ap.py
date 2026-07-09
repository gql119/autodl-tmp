from ue_framework.methods.mtepi import stage2_gate


def test_channel_ablation_ap_gate_requires_real_topk_ap_curve():
    manifest = {"legal_same_trajectory": True}
    ranked = [{"hard_pass": True, "channel_type": "target-selective"}]
    gate = stage2_gate(manifest, ranked, [], [], [], [{"layer": "P3", "channel": 1}], min_target_selective_channels=1, min_authorized_retention=0.9, min_topk_protected_drop=0.1, min_transfer_jaccard=0.5)
    assert gate["gate"] == "FAIL"
    assert any("Top-k AP" in reason for reason in gate["reasons"])
