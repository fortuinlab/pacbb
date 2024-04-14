from torch import nn, Tensor
import torch.nn.functional as F

from core.layer import AbstractProbLayer


class ProbConv2d(nn.Conv2d, AbstractProbLayer):
    def forward(self, input: Tensor) -> Tensor:
        sampled_weight, sampled_bias = self.sample_from_distribution()
        return F.conv2d(input, sampled_weight, sampled_bias, self.stride, self.padding, self.dilation, self.groups)
