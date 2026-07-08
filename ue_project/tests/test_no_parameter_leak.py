from ue_framework.methods.learning_trajectory.virtual_update import parameter_leak_max_abs_diff, snapshot_parameters

from .helpers import make_batch, make_images, make_toy_components


def test_two_virtual_batches_do_not_leak_parameters():
    model, _adapter, method, _config = make_toy_components(seed=10)
    batch = make_batch()
    snapshot = snapshot_parameters(model)

    for seed in [11, 12]:
        support_images = make_images(seed=seed)
        query_images = make_images(seed=seed + 20)
        delta = (support_images * 0.0).requires_grad_()
        result = method.compute_p2_step(support_images, query_images, batch, batch, delta)
        assert result["logs"]["parameter_leak_max_abs_diff"] == 0.0

    assert parameter_leak_max_abs_diff(model, snapshot) == 0.0
