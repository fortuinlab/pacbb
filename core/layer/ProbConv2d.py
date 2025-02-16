import torch.nn.functional as f
from torch import Tensor, nn

from core.layer import AbstractProbLayer


class ProbConv2d(nn.Conv2d, AbstractProbLayer):
    """
    A probabilistic 2D convolution layer.

    Inherits from `nn.Conv2d` and `AbstractProbLayer`. Weights and bias
    are sampled from associated distributions during forward passes.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Perform a 2D convolution using sampled weights and bias.

        Args:
            input (Tensor): The input tensor of shape (N, C_in, H_in, W_in).

        Returns:
            Tensor: The output tensor of shape (N, C_out, H_out, W_out).
        """
        sampled_weight, sampled_bias = self.sample_from_distribution()
        return f.conv2d(
            input,
            sampled_weight,
            sampled_bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
