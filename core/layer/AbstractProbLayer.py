from abc import ABC
from typing import Tuple

from torch import nn, Tensor

from core.distribution import AbstractVariable


class AbstractProbLayer(nn.Module, ABC):
    probabilistic_mode: bool
    _weight_dist: AbstractVariable
    _bias_dist: AbstractVariable
    _prior_weight_dist: AbstractVariable
    _prior_bias_dist: AbstractVariable

    def probabilistic(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("probabilistic mode is expected to be boolean")
        if isinstance(self, AbstractProbLayer):
            self.probabilistic_mode = mode
        for module in self.children():
            if hasattr(module, 'probabilistic_mode') and hasattr(module, 'probabilistic'):
                module.probabilistic(mode)

    def sample_from_distribution(self) -> Tuple[Tensor, Tensor]:
        if self.probabilistic_mode:
            sampled_weight = self._weight_dist.sample()
            sampled_bias = self._bias_dist.sample() if self._bias_dist else None
        else:
            if not self.training:
                sampled_weight = self._weight_dist.mu
                sampled_bias = self._bias_dist.mu if self._bias_dist else None
            else:
                raise ValueError('Only training with probabilistic mode is allowed')
        return sampled_weight, sampled_bias
