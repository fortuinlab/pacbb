from abc import ABC, abstractmethod
from typing import Tuple

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
        device: torch.device,
    ):
        self._kl_penalty = kl_penalty
        self._clamping = clamping
        self._pmin = pmin
        self._num_classes = num_classes
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
        loss_ce, loss_01, outputs = self.compute_losses(model, data, target.long())

        train_obj = self.bound(loss_ce, kl, num_samples)
        return train_obj, kl / num_samples, outputs, loss_ce, loss_01

    def compute_losses(
        self, model: AbstractPBPModel, data: Tensor, targets: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        outputs = model(data, sample=True, clamping=self._clamping, pmin=self._pmin)
        loss_ce = RiskEvaluator.compute_empirical_risk(outputs, targets, self._clamping, self._pmin)
        loss_01 = RiskEvaluator.compute_01_empirical_risk(outputs, targets)
        return loss_ce, loss_01, outputs
