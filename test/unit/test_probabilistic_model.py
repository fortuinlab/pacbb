import torch

from core.model.probabilistic import PBP3Model


def test_probabilistic_pbp3_model():
    input_ = torch.ones((28, 28))
    device = torch.device('cpu')
    model = PBP3Model(28*28, 100, 10, 'gaussian', 0.01, 'zeros', device)

    assert model.kl_div is None
    assert model.compute_kl() >= 1e-6

    model.train(mode=False)
    output = model.forward(input_)
    assert model.l1.kl_div is None
    assert model.l2.kl_div is None
    assert model.l3.kl_div is None

    model.train()
    output = model.forward(input_)
    assert model.l1.kl_div is not None
    assert model.l2.kl_div is not None
    assert model.l3.kl_div is not None

    assert output.shape == torch.Size([1, 10])
