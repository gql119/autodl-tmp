from ue_framework.methods.multitrajectory_gain.feasibility import pearson_correlation, spearman_correlation


def test_proxy_ap_correlation_matches_manual_ordering():
    proxy = [1.0, 2.0, 3.0, 4.0]
    ap = [4.0, 3.0, 2.0, 1.0]
    sp = spearman_correlation(proxy, ap)
    pe = pearson_correlation(proxy, ap)
    assert sp["rho"] == -1.0
    assert pe["r"] == -1.0
