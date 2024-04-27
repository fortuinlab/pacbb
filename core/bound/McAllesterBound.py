from typing import Tuple, Union
import math

from torch import Tensor

from core.bound import AbstractBound
from core.utils.kl import inv_kl


class McAllesterBound(AbstractBound):
    """
        Implements a McAllester PAC Bayes bound.
    """

    def __init__(self, bound_delta: float, loss_delta: float):
        super().__init__(bound_delta, loss_delta)

    def calculate(self,
                  avg_loss: float,
                  kl: Union[Tensor, float],
                  num_samples_bound: int,
                  num_samples_loss: int,
                  ) -> Tuple[Union[Tensor, float], Union[Tensor, float]]:
        """
        Calculates the PAC Bayes bound.

        Args:
            avg_loss (float): The loss averaged using Monte Carlo sampling.
            kl (Union[Tensor, float]): The Kullback-Leibler divergence between prior and posterior distributions.
            num_samples_bound (int): The number of data samples in the bound dataset.
            num_samples_loss (int): The number of Monte Carlo samples.

        Returns:
            Tuple[Union[Tensor, float], Union[Tensor, float]]:
                A tuple containing the calculated PAC Bayes bound and the upper bound of empirical risk.
        """
        empirical_risk = inv_kl(avg_loss, math.log(2 / self._loss_delta) / num_samples_loss)
        risk = empirical_risk + math.sqrt((kl + math.log((2 * math.sqrt(num_samples_bound))) / self._bound_delta) / (2 * num_samples_bound))
        return risk, empirical_risk
