from abc import ABC, abstractmethod
from typing import Tuple
import math

import torch
from torch import Tensor

from core.model.evaluation import RiskEvaluator
from core.model.probabilistic import AbstractPBPModel


class AbstractObjective(ABC):
    def __init__(
        self,
        kl_penalty: float,
        clamping: bool,
        pmin: float,
        num_classes: int,
        num_mc_samples: int,
        delta: float,
        delta_test: float,
        device: torch.device,
    ):
        self._kl_penalty = kl_penalty
        self._clamping = clamping
        self._pmin = pmin
        self._num_classes = num_classes
        self._num_mc_samples = num_mc_samples
        self._delta = delta
        self._delta_test = delta_test
        self._device = device

    @abstractmethod
    def bound(self, empirical_risk: Tensor, kl: Tensor, num_samples: float) -> Tensor:
        pass

    def train_objective(
        self, model: AbstractPBPModel, data, target, num_samples: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # compute train objective and return all metrics
        kl = model.compute_kl(
            recompute=False
        )  # No need to recompute. KL was updated during the last training pass
        loss_ce, loss_01, outputs = self.compute_losses(model, data, target.long(), self._clamping)

        train_obj = self.bound(loss_ce, kl, num_samples)
        return train_obj, kl / num_samples, outputs, loss_ce, loss_01

    def compute_losses(
        self, model: AbstractPBPModel, data: Tensor, targets: Tensor, clamping: bool
    ) -> Tuple[Tensor, Tensor, Tensor]:
        outputs = model(data, sample=True, clamping=clamping, pmin=self._pmin)
        loss_ce = RiskEvaluator.compute_empirical_risk(outputs, targets, clamping, self._pmin)
        loss_01 = RiskEvaluator.compute_01_empirical_risk(outputs, targets)
        return loss_ce, loss_01, outputs

    def _compute_empirical_risk(self, model: AbstractPBPModel, data: Tensor, targets: Tensor, clamping: bool) -> Tuple[float, float]:
        # TODO: change name
        loss_ce_mc = 0.0
        loss_01_mc = 0.0

        for i in range(self._num_mc_samples):
            loss_ce, loss_01, _ = self.compute_losses(model, data, targets.long(), clamping)
            loss_ce_mc += loss_ce
            loss_01_mc += loss_01

        return loss_ce_mc / self._num_mc_samples, loss_01_mc / self._num_mc_samples

    def compute_risks(self, model: AbstractPBPModel, data: Tensor, targets: Tensor, num_samples: int, num_samples_bound: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        kl = model.compute_kl()

        avg_loss_ce, avg_loss_01 = self._compute_empirical_risk(model, data, targets, True)

        empirical_risk_ce = self.inv_kl(avg_loss_ce, math.log(2 / self._delta_test) / self._num_mc_samples)
        empirical_risk_01 = self.inv_kl(avg_loss_01, math.log(2 / self._delta_test) / self._num_mc_samples)

        train_obj = self.bound(empirical_risk_ce, kl, num_samples)

        risk_ce = self.inv_kl(empirical_risk_ce, (kl + math.log((2 * math.sqrt(num_samples_bound)) / self._delta)) / num_samples_bound)
        risk_01 = self.inv_kl(empirical_risk_01, (kl + math.log((2 * math.sqrt(num_samples_bound)) / self._delta)) / num_samples_bound)

        return train_obj, kl/num_samples_bound, empirical_risk_ce, empirical_risk_01, risk_ce, risk_01

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
