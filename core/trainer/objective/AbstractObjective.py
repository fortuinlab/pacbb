from abc import ABC, abstractmethod
from typing import Tuple
import math

import torch
from torch import Tensor

from core.model.evaluation import RiskEvaluator
from core.model.probabilistic import AbstractPBPModel
from core.bound import AbstractBound


class AbstractObjective(ABC):
    def __init__(
        self,
        bound: AbstractBound,
        kl_penalty: float,
        clamping: bool,
        pmin: float,
        num_classes: int,
        num_mc_samples: int,
        device: torch.device,
    ):
        self._kl_penalty = kl_penalty
        self._clamping = clamping
        self._pmin = pmin
        self._num_classes = num_classes
        self._num_mc_samples = num_mc_samples
        self._device = device
        self._bound = bound

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
        loss_ce, loss_01, outputs = RiskEvaluator.compute_losses(model, data, target.long(), self._clamping, self._pmin)
        train_obj = self.bound(loss_ce, kl, num_samples)
        return train_obj, kl / num_samples, outputs, loss_ce, loss_01

    def compute_risks(self, model: AbstractPBPModel, data: Tensor, targets: Tensor, num_samples: int, num_samples_bound: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        kl = model.compute_kl()

        avg_loss_ce, avg_loss_01 = RiskEvaluator.compute_avg_losses(self._num_mc_samples, model, data, targets, True, self._pmin)

        risk_ce, empirical_risk_ce = self._bound.calculate(avg_loss_ce, self._num_mc_samples, kl, num_samples_bound)
        risk_01, empirical_risk_01 = self._bound.calculate(avg_loss_01, self._num_mc_samples, kl, num_samples_bound)

        train_obj = self.bound(empirical_risk_ce, kl, num_samples)

        return train_obj, kl/num_samples_bound, empirical_risk_ce, empirical_risk_01, risk_ce, risk_01
