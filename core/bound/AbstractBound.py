from abc import ABC, abstractmethod

from torch import Tensor


class AbstractBound(ABC):
    """
    Abstract PAC bound class for evaluating risk certificates.

    Args:
        bound_delta (float): Confidence level over random data samples.
            It represents the probability that the upper bound of the PAC bound holds.
        loss_delta (float): Confidence level over random weight samples.
            It represents the probability that the upper bound of empirical loss holds.

    Overall probability is (1 - loss_bound) - bound_delta.

    Attributes:
        _bound_delta (float): Confidence level over random data samples.
        _loss_delta (float): Confidence level over random weight samples.
    """

    def __init__(self, bound_delta: float, loss_delta: float):
        self._bound_delta = bound_delta
        self._loss_delta = loss_delta

    @abstractmethod
    def calculate(self, *args, **kwargs) -> tuple[Tensor | float, Tensor | float]:
        """
        Calculates the PAC Bayes bound.

        Args:
            args: Variable length argument list.
            kwargs: Arbitrary keyword arguments.

        Returns:
            Tuple[Union[Tensor, float], Union[Tensor, float]]:
                A tuple containing the calculated PAC bound and the upper bound of empirical risk.
        """
        pass
