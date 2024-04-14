from torch import nn, Tensor
import torch.nn.functional as F

from core.layer import AbstractProbLayer


class ProbLinear(nn.Linear, AbstractProbLayer):
    def forward(self, input: Tensor) -> Tensor:
        sampled_weight, sampled_bias = self.sample_from_distribution()
        return F.linear(input, sampled_weight, sampled_bias)
