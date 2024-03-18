from typing import List, Union

import torch
from torch import nn, Tensor
from app.model.probabilistic.distribution import AbstractVariable, GaussianVariable, LaplaceVariable
from app.model import ModelUtils


class VariableUtils:
    @staticmethod
    def from_flat(model: nn.Module, rho: Union[Tensor, List[float]], distribution: str, device: torch.device) -> List[AbstractVariable]:
        # TODO: replace with Factory
        if distribution == 'gaussian':
            distribution = GaussianVariable
        elif distribution == 'laplace':
            distribution = LaplaceVariable
        else:
            raise ValueError(f"Unknown distribution {distribution}")

        distributions = []
        shift = 0
        for layer in ModelUtils.get_layers(model):
            weight_cutoff = shift + layer.out_features * layer.in_features
            bias_cutoff = weight_cutoff + layer.out_features
            weight_distribution = distribution(mu=layer.weight,
                                               rho=rho[shift: weight_cutoff].reshape(layer.out_features, layer.in_features),
                                               device=device,
                                               fix_mu=False,
                                               fix_rho=False)
            bias_distribution = distribution(mu=layer.bias,
                                             rho=rho[weight_cutoff: bias_cutoff],
                                             device=device,
                                             fix_mu=False,
                                             fix_rho=False)
            distributions.append(weight_distribution)
            distributions.append(bias_distribution)
            shift = bias_cutoff
        return distributions

    @staticmethod
    def from_pbp(model: torch.nn.Module) -> List[AbstractVariable]:
        distributions = []
        for layer in ModelUtils.get_pbp_layers(model):
            distributions.append(layer.weight)
            distributions.append(layer.bias)
        return distributions

    @staticmethod
    def from_bayesian(model: torch.nn.Module, distribution: str, device: torch.device, fix: bool = False) -> List[AbstractVariable]:
        if distribution == 'gaussian':
            distribution = GaussianVariable
        elif distribution == 'laplace':
            distribution = LaplaceVariable
        else:
            raise ValueError(f"Unknown distribution {distribution}")

        distributions = []
        for layer in model.children():
            if hasattr(layer, 'mu_weight') and hasattr(layer, 'rho_weight') and hasattr(layer, 'mu_bias') and hasattr(layer, 'rho_bias'):
                distributions.append(distribution(mu=layer.mu_weight,
                                                  rho=layer.rho_weight,
                                                  device=device,
                                                  fix_mu=fix, fix_rho=fix))
                distributions.append(distribution(mu=layer.mu_bias,
                                                  rho=layer.rho_bias,
                                                  device=device,
                                                  fix_mu=fix, fix_rho=fix))
        return distributions
