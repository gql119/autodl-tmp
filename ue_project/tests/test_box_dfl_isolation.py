import torch

from tests.test_supervision_decomposer import make_decomposition


def test_person_box_dfl_intervention_changes_only_protected():
    before = make_decomposition()
    after = make_decomposition(
        per_box=torch.tensor([[0.9, 0.3, 0.0]]),
        per_dfl=torch.tensor([[0.8, 0.5, 0.0]]),
    )
    assert (after.protected_box - before.protected_box).abs().item() > 1.0e-5
    assert (after.protected_dfl - before.protected_dfl).abs().item() > 1.0e-5
    assert (after.authorized_box - before.authorized_box).abs().item() < 1.0e-6
    assert (after.authorized_dfl - before.authorized_dfl).abs().item() < 1.0e-6


def test_authorized_box_dfl_intervention_changes_only_authorized():
    before = make_decomposition()
    after = make_decomposition(
        per_box=torch.tensor([[0.2, 0.9, 0.0]]),
        per_dfl=torch.tensor([[0.4, 0.9, 0.0]]),
    )
    assert (after.authorized_box - before.authorized_box).abs().item() > 1.0e-5
    assert (after.authorized_dfl - before.authorized_dfl).abs().item() > 1.0e-5
    assert (after.protected_box - before.protected_box).abs().item() < 1.0e-6
    assert (after.protected_dfl - before.protected_dfl).abs().item() < 1.0e-6


def test_ambiguous_box_dfl_intervention_changes_only_shared():
    ambiguous = torch.tensor([[True, False, False]])
    before = make_decomposition(ambiguous_mask=ambiguous)
    after = make_decomposition(
        ambiguous_mask=ambiguous,
        per_box=torch.tensor([[0.9, 0.3, 0.0]]),
        per_dfl=torch.tensor([[0.8, 0.5, 0.0]]),
    )
    assert (after.shared_box - before.shared_box).abs().item() > 1.0e-5
    assert (after.shared_dfl - before.shared_dfl).abs().item() > 1.0e-5
    assert after.protected_box.item() == 0.0
    assert after.protected_dfl.item() == 0.0
    assert (after.authorized_box - before.authorized_box).abs().item() < 1.0e-6
