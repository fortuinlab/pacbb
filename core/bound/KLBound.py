from typing import Tuple, Union
import math

from torch import Tensor
from core.bound import AbstractBound


class KLBound(AbstractBound):
    def __init__(self, delta: float, delta_test: float):
        super().__init__(delta)
        self._delta_test = delta_test

    def calculate(self, avg_loss: float, num_mc_samples: int, kl: Union[Tensor, float], num_samples_bound: int) -> Tuple[Tensor, Tensor]:
        empirical_risk = self.inv_kl(avg_loss, math.log(2 / self._delta_test) / num_mc_samples)

        risk = self.inv_kl(empirical_risk,
                              (kl + math.log((2 * math.sqrt(num_samples_bound)) / self._delta)) / num_samples_bound)

        return risk, empirical_risk

    @staticmethod
    def inv_kl(qs, ks):
        """Inversion of the binary kl

        Parameters
        ----------
        qs : float
            Empirical risk

        ks : float
            second term for the binary kl inversion

        """
        # TODO: refactor
        # computation of the inversion of the binary KL
        ikl = 0
        izq = qs
        dch = 1 - 1e-10
        while True:
            p = (izq + dch) * .5
            if qs == 0:
                ikl = ks - (0 + (1 - qs) * math.log((1 - qs) / (1 - p)))
            elif qs == 1:
                ikl = ks - (qs * math.log(qs / p) + 0)
            else:
                ikl = ks - (qs * math.log(qs / p) + (1 - qs) * math.log((1 - qs) / (1 - p)))
            if ikl < 0:
                dch = p
            else:
                izq = p
            if (dch - izq) / dch < 1e-5:
                break
        return p
