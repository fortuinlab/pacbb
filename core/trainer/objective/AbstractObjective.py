from abc import ABC, abstractmethod
from typing import Union, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from core.model.probabilistic import AbstractPBPModel


class AbstractObjective(ABC):
    def __init__(self, kl_penalty: float, clamping: bool, pmin: float, num_classes: int, device: torch.device):
        self._kl_penalty = kl_penalty
        self._clamping = clamping
        self._pmin = pmin
        self._num_classes = num_classes
        self._device = device

    @abstractmethod
    def bound(self, empirical_risk: float, kl: Tensor, num_samples: float) -> Tensor:
        pass

    def train_objective(self, model: AbstractPBPModel, data, target, num_samples: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # compute train objective and return all metrics
        kl = model.compute_kl()
        loss_ce, loss_01, outputs = self.compute_losses(model, data, target.long())

        train_obj = self.bound(loss_ce, kl, num_samples)
        return train_obj, kl/num_samples, outputs, loss_ce, loss_01

    def compute_losses(self, model: AbstractPBPModel, data: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        outputs = model(data, sample=True, clamping=self._clamping, pmin=self._pmin)
        loss_ce = self.compute_empirical_risk(outputs, targets, bounded=self._clamping)
        loss_01 = self.compute_empirical_risk(outputs, targets)
        return loss_ce, loss_01, outputs

    def compute_empirical_risk(self, outputs: Tensor, targets: Tensor, bounded=True) -> Tensor:
        empirical_risk = F.nll_loss(outputs, targets)
        if bounded:
            empirical_risk = empirical_risk / (np.log(1. / self._pmin))
        return empirical_risk

    def compute_01_empirical_risk(self, outputs: Tensor, targets: Tensor) -> Tensor:
        predictions = outputs.max(1, keepdim=True)[1]
        correct = predictions.eq(targets.view_as(predictions)).sum().item()
        total = targets.size(0)
        loss_01 = 1 - (correct / total)
        return loss_01
