import torch

from tests.test_supervision_decomposer import make_decomposition


def _components(pred_scores):
    dec = make_decomposition(pred_scores=pred_scores)
    return dec.protected_cls.detach(), dec.authorized_cls.detach(), dec.shared_cls.detach()


def _assert_changed_only(before, after, changed_index):
    for idx, (old, new) in enumerate(zip(before, after)):
        delta = (new - old).abs().item()
        if idx == changed_index:
            assert delta > 1.0e-5
        else:
            assert delta < 1.0e-6


def test_person_assigned_class_logit_changes_only_protected_cls():
    pred = torch.zeros((1, 3, 20))
    before = _components(pred)
    pred[0, 0, 14] = 2.0
    after = _components(pred)
    _assert_changed_only(before, after, 0)


def test_authorized_assigned_class_logit_changes_only_authorized_cls():
    pred = torch.zeros((1, 3, 20))
    before = _components(pred)
    pred[0, 1, 1] = 2.0
    after = _components(pred)
    _assert_changed_only(before, after, 1)


def test_person_positive_other_class_logit_changes_only_shared_cls():
    pred = torch.zeros((1, 3, 20))
    before = _components(pred)
    pred[0, 0, 2] = 2.0
    after = _components(pred)
    _assert_changed_only(before, after, 2)


def test_authorized_positive_person_logit_changes_only_shared_cls():
    pred = torch.zeros((1, 3, 20))
    before = _components(pred)
    pred[0, 1, 14] = 2.0
    after = _components(pred)
    _assert_changed_only(before, after, 2)


def test_background_logits_change_only_shared_cls():
    pred = torch.zeros((1, 3, 20))
    before = _components(pred)
    pred[0, 2, :] = 2.0
    after = _components(pred)
    _assert_changed_only(before, after, 2)
