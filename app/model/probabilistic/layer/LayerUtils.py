import math

import torch


class LayerUtils:
    @staticmethod
    def compute_standard_normal_cdf(x: float) -> float:
        """
        Compute the standard normal cumulative distribution function.

        Parameters:
        x (float): The input value.

        Returns:
        float: The cumulative distribution function value at x.
        """
        # TODO: replace with numpy
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    @staticmethod
    def truncated_normal_fill_tensor(
        tensor: torch.Tensor,
        mean: float = 0.0,
        std: float = 1.0,
        a: float = -2.0,
        b: float = 2.0,
    ) -> torch.Tensor:
        # TODO: refactor
        with torch.no_grad():
            # Get upper and lower cdf values
            l = LayerUtils.compute_standard_normal_cdf((a - mean) / std)
            u = LayerUtils.compute_standard_normal_cdf((b - mean) / std)

            # Fill tensor with uniform values from [l, u]
            tensor.uniform_(l, u)

            # Use inverse cdf transform from normal distribution
            tensor.mul_(2)
            tensor.sub_(1)

            # Ensure that the values are strictly between -1 and 1 for erfinv
            eps = torch.finfo(tensor.dtype).eps
            tensor.clamp_(min=-(1.0 - eps), max=(1.0 - eps))
            tensor.erfinv_()

            # Transform to proper mean, std
            tensor.mul_(std * math.sqrt(2.0))
            tensor.add_(mean)

            # Clamp one last time to ensure it's still in the proper range
            tensor.clamp_(min=a, max=b)
            return tensor
