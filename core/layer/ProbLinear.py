import torch.nn.functional as f
from torch import Tensor, nn

from core.layer import AbstractProbLayer


class ProbLinear(nn.Linear, AbstractProbLayer):
    """
    A probabilistic linear (fully connected) layer.

    Extends `nn.Linear` such that weights and bias are sampled from
    a distribution during each forward pass if `probabilistic_mode` is True.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for a probabilistic linear layer.

        Args:
            input (Tensor): Input tensor of shape (N, in_features).

        Returns:
            Tensor: Output tensor of shape (N, out_features).
        """
        sampled_weight, sampled_bias = self.sample_from_distribution()
        return f.linear(input, sampled_weight, sampled_bias)
