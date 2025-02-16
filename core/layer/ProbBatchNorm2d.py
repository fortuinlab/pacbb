import torch.nn.functional as f
from torch import Tensor, nn

from core.layer import AbstractProbLayer


class ProbBatchNorm2d(nn.BatchNorm2d, AbstractProbLayer):
    """
    A probabilistic 2D batch normalization layer.

    Extends PyTorch's `nn.BatchNorm2d` to sample weight and bias from learned
    distributions for use in a probabilistic neural network.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for probabilistic 2D batch normalization.

        During training:
          - Uses mini-batch statistics to normalize.
          - Samples weight and bias if `probabilistic_mode` is True.

        Args:
            input (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Batch-normalized output of the same shape as `input`.
        """
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        sampled_weight, sampled_bias = self.sample_from_distribution()

        return f.batch_norm(
            input,
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            sampled_weight,
            sampled_bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
