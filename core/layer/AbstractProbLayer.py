from abc import ABC
from typing import Tuple

from torch import nn, Tensor

from core.distribution import AbstractVariable


class AbstractProbLayer(nn.Module, ABC):
    """
    Base class for probabilistic layers in a neural network.

    It introduces the concept of `probabilistic_mode`, which determines whether
    a layer samples parameters from a distribution or uses deterministic values.
    Each layer holds references to distributions for weights/biases and their prior.
    """
    probabilistic_mode: bool
    _weight_dist: AbstractVariable
    _bias_dist: AbstractVariable
    _prior_weight_dist: AbstractVariable
    _prior_bias_dist: AbstractVariable

    def probabilistic(self, mode: bool = True):
        """
        Recursively set the `probabilistic_mode` flag for this layer and its children.

        Args:
            mode (bool, optional): If True, the layer will draw samples from its
                distributions during forward passes. If False, it will typically
                use deterministic parameters (i.e., the mean), except in training.
        """
        if not isinstance(mode, bool):
            raise ValueError("probabilistic mode is expected to be boolean")
        if isinstance(self, AbstractProbLayer):
            self.probabilistic_mode = mode
        for module in self.children():
            if hasattr(module, 'probabilistic_mode') and hasattr(module, 'probabilistic'):
                module.probabilistic(mode)

    def sample_from_distribution(self) -> Tuple[Tensor, Tensor]:
        """
        Draw samples for weight and bias from their respective distributions.

        Behavior depends on `probabilistic_mode`:
          - If `probabilistic_mode` is True, the method samples from `_weight_dist`
            and `_bias_dist`.
          - If `probabilistic_mode` is False and the layer is in eval mode, it uses
            the mean (mu) of each distribution.
          - If `probabilistic_mode` is False and the layer is in training mode, it
            raises an error, as training requires sampling.

        Returns:
            Tuple[Tensor, Tensor]: Sampled weight and bias. If the layer has no bias,
            the second returned value is None.
        """
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
