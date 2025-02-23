import torch.nn.functional as f
from torch import Tensor, nn

from core.layer import AbstractProbLayer


class ProbBatchNorm1d(nn.BatchNorm1d, AbstractProbLayer):
    """
    A probabilistic 1D batch normalization layer.

    This layer extends PyTorch's `nn.BatchNorm1d` to sample weight and bias
    from learned distributions. The forward pass behavior is the same as standard
    batch norm, except the parameters come from `sample_from_distribution`.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for probabilistic batch normalization.

        During training:
          - Maintains running statistics if `track_running_stats` is True.
          - Samples weight/bias if `probabilistic_mode` is True.

        Args:
            input (Tensor): Input tensor of shape (N, C, L) or (N, C).

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
