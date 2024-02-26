import torch

from core.model.probabilistic.distribution import GaussianVariable


def test_gaussian_distribution():
    size = 10
    mu = torch.zeros(size)
    scaled_mu = torch.zeros(size).mul_(3)
    sigma = torch.ones(size)
    scaled_sigma = torch.ones(size).mul_(3)
    rho = torch.log(torch.exp(sigma) - 1.0)
    scaled_rho = torch.log(torch.exp(scaled_sigma) - 1.0)
    device = torch.device("cpu")
    distribution1 = GaussianVariable(mu, rho, device, False, False)
    distribution2 = GaussianVariable(scaled_mu, rho, device, True, True)
    distribution3 = GaussianVariable(mu, scaled_rho, device, True, False)
    distribution4 = GaussianVariable(scaled_mu, scaled_sigma, device, False, True)

    assert distribution1.compute_kl(distribution1).item() <= 1e-6
    assert distribution2.compute_kl(distribution1).item() <= 1e-6
    assert distribution3.compute_kl(distribution1).item() >= 1e-6
    assert (
        distribution3.compute_kl(distribution1).item()
        - distribution3.compute_kl(distribution2).item()
        <= 1e-6
    )
    assert (
        distribution4.compute_kl(distribution1).item()
        - distribution3.compute_kl(distribution2).item()
        >= 1e-6
    )
