import torch

from tests.test_j3_rollout import make_engine, make_sequence


def test_rollout_records_dynamic_assignment_diagnostics_each_step():
    engine = make_engine(steps=3)
    delta = torch.zeros((1, 3, 8, 8), requires_grad=True)
    out = engine.run(make_sequence(), delta, create_graph=True)
    assert len(out.per_step) == 3
    for step in out.per_step:
        assert "clean_protected_positive_count" in step
        assert "poison_protected_positive_count" in step
        assert "assignment_overlap" in step
