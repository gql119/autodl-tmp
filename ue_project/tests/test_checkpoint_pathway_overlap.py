from ue_framework.methods.mtepi import cross_checkpoint_transfer_matrix


def test_checkpoint_pathway_overlap_uses_same_layer_channel_indices():
    rankings = {
        "early": [{"layer": "P3", "channel": 1, "hard_pass": True}, {"layer": "P3", "channel": 2, "hard_pass": True}],
        "middle": [{"layer": "P3", "channel": 1, "hard_pass": True}, {"layer": "P4", "channel": 2, "hard_pass": True}],
    }
    rows = cross_checkpoint_transfer_matrix(rankings, top_k=2)
    off = [r for r in rows if r["source"] == "early" and r["target"] == "middle"][0]
    assert off["topk_jaccard"] == 1.0 / 3.0
