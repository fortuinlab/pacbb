import copy
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.probabilistic.distribution import (AbstractVariable,
                                                   GaussianVariable,
                                                   LaplaceVariable)
from core.model.probabilistic.layer import LayerUtils


class ProbabilisticLinearLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model_weight_distribution: str,
        sigma: float,
        weight_initialization_method: str,
        device: torch.device,
    ):
        super().__init__()
        self._model_weight_distribution = model_weight_distribution
        self._sigma = sigma
        self._weight_initialization_method = weight_initialization_method
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._device = device
        self._rho = np.log(np.exp(sigma) - 1.0)
        # TODO: set as tensor of -1?
        self.kl_div = None

        distribution = self._select_distribution()
        self._initialize_prior(distribution)
        self._initialize_posterior(distribution)

    def _initialize_posterior(self, distribution: Type[AbstractVariable]) -> None:
        # TODO: move initialization to separate functions
        w = 1 / np.sqrt(self._input_dim)
        if self._weight_initialization_method in ["zeros", "random"]:
            bias_mu_posterior = torch.zeros(self._output_dim)
            bias_rho_posterior = torch.ones(self._output_dim) * self._rho
            t = torch.Tensor(self._output_dim, self._input_dim)
            weights_mu_posterior = LayerUtils.truncated_normal_fill_tensor(
                t, 0, w, -2 * w, 2 * w
            )
            weights_rho_posterior = (
                torch.ones(self._output_dim, self._input_dim) * self._rho
            )
        else:
            raise RuntimeError(
                f"Invalid value of weight_initialization_method: {self._weight_initialization_method}"
            )

        self.bias = distribution(
            bias_mu_posterior.clone(),
            bias_rho_posterior.clone(),
            device=self._device,
            fix_mu=False,
            fix_rho=False,
        )
        self.weight = distribution(
            weights_mu_posterior.clone(),
            weights_rho_posterior.clone(),
            device=self._device,
            fix_mu=False,
            fix_rho=False,
        )

    def _initialize_prior(self, distribution: Type[AbstractVariable]) -> None:
        # TODO: move initialization to separate functions
        w = 1 / np.sqrt(self._input_dim)
        if self._weight_initialization_method == "zeros":
            bias_mu_prior = torch.zeros(self._output_dim)
            bias_rho_prior = torch.ones(self._output_dim) * self._rho
            weights_mu_prior = torch.zeros(self._output_dim, self._input_dim)
            weights_rho_prior = (
                torch.ones(self._output_dim, self._input_dim) * self._rho
            )
        elif self._weight_initialization_method == "random":
            bias_mu_prior = torch.zeros(self._output_dim)
            bias_rho_prior = torch.ones(self._output_dim) * self._rho
            t = torch.Tensor(self._output_dim, self._input_dim)
            weights_mu_prior = LayerUtils.truncated_normal_fill_tensor(
                t, 0, w, -2 * w, 2 * w
            )
            weights_rho_prior = (
                torch.ones(self._output_dim, self._input_dim) * self._rho
            )
        else:
            raise RuntimeError(
                f"Invalid value of weight_initialization_method: {self._weight_initialization_method}"
            )

        self.weight_prior = distribution(
            weights_mu_prior.clone(),
            weights_rho_prior.clone(),
            device=self._device,
            fix_mu=True,
            fix_rho=True,
        )
        self.bias_prior = distribution(
            bias_mu_prior.clone(),
            bias_rho_prior.clone(),
            device=self._device,
            fix_mu=True,
            fix_rho=True,
        )

    def _select_distribution(self) -> Type[AbstractVariable]:
        # TODO: move to factory?
        if self._model_weight_distribution == "gaussian":
            distribution = GaussianVariable
        elif self._model_weight_distribution == "laplace":
            distribution = LaplaceVariable
        else:
            raise RuntimeError(
                f"Invalid value of model_weight_distribution: {self._model_weight_distribution}"
            )
        return distribution

    def forward(self, input, force_sampling: bool = False):
        if self.training or force_sampling:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the mean
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            self.kl_div = self.compute_kl()
        return F.linear(input, weight, bias)

    def compute_kl(self, recompute: bool = True) -> torch.Tensor:
        if recompute:
            return self.weight.compute_kl(self.weight_prior) + self.bias.compute_kl(
                self.bias_prior
            )
        else:
            return self.weight.kl_div + self.bias.kl_div

    def __deepcopy__(self, memo):
        # TODO: refactor
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None
        # TODO: does it forget device?
        self.kl_div = self.kl_div.detach().clone()
        copy_ = copy.deepcopy(self, memo)
        self.__deepcopy__ = deepcopy_method
        copy_.__deepcopy__ = deepcopy_method
        return copy_

    def set_weights_from_layer(self, layer: 'ProbabilisticLinearLayer') -> None:
        distribution = layer._select_distribution()

        weight_mu = layer.weight.mu
        weight_rho = layer.weight.rho
        bias_mu = layer.bias_prior.mu
        bias_rho = layer.bias_prior.rho

        self.weight_prior = distribution(weight_mu, weight_rho, self._device, True, True)
        self.bias_prior = distribution(bias_mu, bias_rho, self._device, True, True)

        self.weight = distribution(weight_mu, weight_rho, self._device, False, False)
        self.bias = distribution(bias_mu, bias_rho, self._device, False, False)
