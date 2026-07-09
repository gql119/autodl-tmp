from ue_framework.methods.mtepi import build_consensus_pathways


def test_consensus_pathway_selection_intersects_hard_pass_topk():
    rankings = {
        "early": [{"layer": "P3", "channel": 1, "hard_pass": True, "diff_score": 2.0}],
        "middle": [{"layer": "P3", "channel": 1, "hard_pass": True, "diff_score": 1.0}],
        "late": [{"layer": "P3", "channel": 2, "hard_pass": True, "diff_score": 3.0}],
    }
    assert build_consensus_pathways(rankings, top_k=1) == []
    rankings["late"] = [{"layer": "P3", "channel": 1, "hard_pass": True, "diff_score": 3.0}]
    assert build_consensus_pathways(rankings, top_k=1)[0]["channel"] == 1
