import torch

from ue_framework.core.localized_support import LocalizedSupportBuilder
from ue_framework.methods.learning_trajectory.virtual_update import functional_forward
from ue_framework.methods.multitrajectory_gain import BatchData, J3RolloutEngine, TrajectoryBatchSequence


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 3)

    def forward(self, images):
        return self.head(images.mean(dim=(2, 3)))


class TinyAdapter:
    def __init__(self):
        self.model = TinyModel()

    def get_named_trainable_parameters(self, scope):
        return [(name, param) for name, param in self.model.named_parameters()]

    def forward_with_parameters(self, images, params):
        return functional_forward(self.model, params, images)

    def compute_detection_loss(self, predictions, batch, class_filter=None, return_components=False):
        target = batch["target"].to(predictions)
        loss = torch.nn.functional.mse_loss(predictions, target, reduction="sum")
        out = {"total_loss": loss, "cls_loss": loss, "box_loss": loss * 0.0, "dfl_loss": loss * 0.0}
        return out if return_components else loss


class TinyDecomposer:
    def decompose(self, predictions, batch):
        class Obj:
            pass

        target = batch["target"].to(predictions)
        losses = (predictions - target).pow(2).sum(dim=0)
        obj = Obj()
        obj.protected_total = losses[0]
        obj.authorized_total = losses[1]
        obj.shared_total = losses[2]
        obj.statistics = {
            "protected_positive_count": 1.0,
            "authorized_positive_count": 1.0,
            "shared_positive_count": 1.0,
            "background_count": 1.0,
            "ambiguous_positive_count": 0.0,
        }
        return obj


def make_sequence():
    support = []
    for idx in range(3):
        images = torch.full((1, 3, 8, 8), 0.1 + idx * 0.1)
        batch = {
            "target": torch.tensor([[1.0, 1.0, 1.0]]),
            "cls": torch.tensor([14.0]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            "batch_idx": torch.tensor([0.0]),
            "batch_size": 1,
        }
        support.append(BatchData(images, batch, [f"s{idx}"]))
    query = BatchData(
        torch.full((1, 3, 8, 8), 0.6),
        {
            "target": torch.tensor([[1.0, 1.0, 1.0]]),
            "cls": torch.tensor([14.0]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            "batch_idx": torch.tensor([0.0]),
            "batch_size": 1,
        },
        ["q0"],
    )
    return TrajectoryBatchSequence(support, query, [b.image_ids for b in support], query.image_ids, 0, 0)


def make_engine(steps=3):
    return J3RolloutEngine(
        TinyAdapter(),
        TinyDecomposer(),
        LocalizedSupportBuilder(protected_class_id=14),
        steps=steps,
        learning_rate=0.1,
        momentum=0.0,
        weight_decay=0.0,
        lambda_regularization=0.0,
    )


def test_j3_rollout_executes_three_steps_and_reaches_delta():
    torch.manual_seed(0)
    engine = make_engine(steps=3)
    delta = torch.zeros((1, 3, 8, 8), requires_grad=True)
    out = engine.run(make_sequence(), delta, create_graph=True)
    grad = torch.autograd.grad(out.loss, delta, allow_unused=False)[0]
    assert len(out.per_step) == 3
    assert out.logs["steps_executed"] == 3.0
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0.0


def test_j3_rollout_does_not_leak_parameters():
    engine = make_engine(steps=3)
    before = {name: param.detach().clone() for name, param in engine.adapter.model.named_parameters()}
    delta = torch.zeros((1, 3, 8, 8), requires_grad=True)
    out = engine.run(make_sequence(), delta, create_graph=True)
    assert out.logs["surrogate_parameter_max_abs_diff"] == 0.0
    for name, param in engine.adapter.model.named_parameters():
        assert torch.allclose(param.detach(), before[name])
