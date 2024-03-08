import torch
from torch import Tensor

from app.model import PBP3Model


def test_probabilistic_pbp3_model():
    input_ = torch.ones((28, 28))
    device = torch.device("cpu")
    model = PBP3Model(28 * 28, 100, 10, "gaussian", 0.01, "zeros", device)

    assert model.kl_div is None
    assert model.compute_kl() >= 1e-6

    model.train(mode=False)
    output = model.forward(input_)
    assert model.l1.kl_div.eq(Tensor([0]))
    assert model.l2.kl_div.eq(Tensor([0]))
    assert model.l3.kl_div.eq(Tensor([0]))

    model.train()
    output = model.forward(input_)
    assert model.l1.kl_div is not None
    assert model.l2.kl_div is not None
    assert model.l3.kl_div is not None

    assert output.shape == torch.Size([1, 10])

    model2 = PBP3Model(28 * 28, 100, 10, "gaussian", 0.01, "random", device)
    model2.set_weights_from_model(model)

    assert model.l1.weight.compute_kl(model2.l1.weight) <= 1e-6
    assert model.l2.weight.compute_kl(model2.l2.weight) <= 1e-6
    assert model.l3.weight.compute_kl(model2.l3.weight) <= 1e-6
