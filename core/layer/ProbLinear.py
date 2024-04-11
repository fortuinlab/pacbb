from torch import nn, Tensor
import torch.nn.functional as F

from core.layer import AbstractProbLayer


class ProbLinear(nn.Linear, AbstractProbLayer):
    def forward(self, input: Tensor) -> Tensor:
        if self.probabilistic_mode:
            sampled_weight = self._weight_dist.sample()
            sampled_bias = self._bias_dist.sample() if self._bias_dist else None
        else:
            if not self.training:
                sampled_weight = self._weight_dist.mu
                sampled_bias = self._bias_dist.mu if self._bias_dist else None
            else:
                raise ValueError('Only training with probabilistic mode is allowed')
        return F.linear(input, sampled_weight, sampled_bias)
