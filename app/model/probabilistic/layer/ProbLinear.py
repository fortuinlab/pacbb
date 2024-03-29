from torch import nn, Tensor
import torch.nn.functional as F


class ProbLinear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight_dist.sample()
        bias = self.bias_dist.sample()
        return F.linear(input, weight, bias)
