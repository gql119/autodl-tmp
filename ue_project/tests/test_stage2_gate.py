from ue_framework.methods.mtepi import stage2_gate


def test_stage2_gate_fails_without_legal_same_trajectory_checkpoints():
    gate = stage2_gate(
        {"legal_same_trajectory": False},
        [{"hard_pass": True, "channel_type": "target-selective"}],
        [{"protected_ap_drop": 0.2, "authorized_retention": 0.95}],
        [{"source": "early", "target": "middle", "topk_jaccard": 0.8}],
        [{"mean": 1.0}],
        [{"layer": "P3", "channel": 1}],
        min_target_selective_channels=1,
        min_authorized_retention=0.9,
        min_topk_protected_drop=0.1,
        min_transfer_jaccard=0.5,
    )
    assert gate["gate"] == "FAIL"
    assert any("same-trajectory" in reason for reason in gate["reasons"])
