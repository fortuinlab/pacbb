import torch
from torch import Tensor

from app.model import ProbabilisticLinearLayer


def test_probabilistic_linear_layer():
    input_ = torch.ones((28, 28)).flatten()
    device = torch.device("cpu")
    layer1 = ProbabilisticLinearLayer(28 * 28, 10, "gaussian", 0.01, "zeros", device)

    assert layer1.kl_div.eq(Tensor([0]))
    assert layer1.compute_kl() >= 1e-6
    assert torch.allclose(
        layer1.weight_prior.mu, torch.zeros_like(layer1.weight_prior.mu)
    )
    assert torch.allclose(
        layer1.weight_prior.rho, torch.ones_like(layer1.weight_prior.rho) * layer1._rho
    )
    assert torch.allclose(layer1.bias_prior.mu, torch.zeros_like(layer1.bias_prior.mu))
    assert torch.allclose(
        layer1.bias_prior.rho, torch.ones_like(layer1.bias_prior.rho) * layer1._rho
    )
    assert torch.allclose(layer1.bias.mu, torch.zeros_like(layer1.bias.mu))
    assert torch.allclose(
        layer1.bias.rho, torch.ones_like(layer1.bias.rho) * layer1._rho
    )

    layer1.train(mode=False)
    output = layer1.forward(input_)
    assert layer1.kl_div.eq(Tensor([0]))

    layer1.train()
    output = layer1.forward(input_)
    assert layer1.kl_div is not None

    assert output.shape == torch.Size([10])
