import numpy as np
import torch.nn.functional as F
from torch import Tensor
import torch
from torch import nn
from typing import Tuple, List
from torch.utils import data

from app.model.ModelUtils import ModelUtils
from app.model.probabilistic import AbstractPBPModel
from app.model.probabilistic.distribution import AbstractVariable



class RiskEvaluator:
    @staticmethod
    def compute_ce_empirical_loss(
        outputs: Tensor, targets: Tensor, clamping: bool, pmin: float
    ) -> Tensor:
        empirical_risk = F.nll_loss(outputs, targets)
        if clamping:
            empirical_risk = empirical_risk / (np.log(1.0 / pmin))
        return empirical_risk

    @staticmethod
    def compute_01_empirical_loss(outputs: Tensor, targets: Tensor) -> Tensor:
        predictions = outputs.max(1, keepdim=True)[1]
        correct = predictions.eq(targets.view_as(predictions)).sum().item()
        total = targets.size(0)
        loss_01 = 1 - (correct / total)
        return Tensor([loss_01])

    @staticmethod
    def compute_losses(model: nn.Module, distribution: List[AbstractVariable], data: Tensor, targets: Tensor, clamping: bool, pmin: float) -> Tuple[Tensor, Tensor, Tensor]:
        if isinstance(model, AbstractPBPModel):
            outputs = model.forward(data, True, clamping, pmin)
        else:
            outputs = ModelUtils.sample_weight_call(model, distribution, data)
            if clamping:
                outputs = torch.clamp(outputs, np.log(pmin))
        loss_ce = RiskEvaluator.compute_ce_empirical_loss(outputs, targets, clamping, pmin)
        loss_01 = RiskEvaluator.compute_01_empirical_loss(outputs, targets)
        return loss_ce, loss_01, outputs

    @staticmethod
    def compute_losses_old(model: nn.Module, data: Tensor, targets: Tensor,
                       clamping: bool, pmin: float) -> Tuple[Tensor, Tensor, Tensor]:
        if isinstance(model, AbstractPBPModel):
            outputs = model.forward(data, True, clamping, pmin)
        else:
            raise ValueError()
        loss_ce = RiskEvaluator.compute_ce_empirical_loss(outputs, targets, clamping, pmin)
        loss_01 = RiskEvaluator.compute_01_empirical_loss(outputs, targets)
        return loss_ce, loss_01, outputs

    @staticmethod
    def compute_avg_losses_old(num_mc_samples: int,
                           model: nn.Module,
                           data: Tensor,
                           targets: Tensor,
                           clamping: bool,
                           pmin: float) -> Tuple[float, float]:
        loss_ce_mc = 0.0
        loss_01_mc = 0.0

        for i in range(num_mc_samples):
            loss_ce, loss_01, _ = RiskEvaluator.compute_losses_old(model, data, targets.long(), clamping, pmin)
            loss_ce_mc += loss_ce
            loss_01_mc += loss_01

        return loss_ce_mc / num_mc_samples, loss_01_mc / num_mc_samples

    @staticmethod
    def compute_avg_losses(num_mc_samples: int,
                           model: nn.Module,
                           distribution: List[AbstractVariable],
                           data: Tensor,
                           targets: Tensor,
                           clamping: bool,
                           pmin: float) -> Tuple[float, float]:
        loss_ce_mc = 0.0
        loss_01_mc = 0.0

        for i in range(num_mc_samples):
            loss_ce, loss_01, _ = RiskEvaluator.compute_losses(model, distribution, data, targets.long(), clamping, pmin)
            loss_ce_mc += loss_ce
            loss_01_mc += loss_01

        return loss_ce_mc / num_mc_samples, loss_01_mc / num_mc_samples

    @staticmethod
    def compute_empirical_losses(num_mc_samples: int,
                                 model: nn.Module,
                                 distribution: List[AbstractVariable],
                                 loader: data.dataloader.DataLoader,
                                 clamping: bool,
                                 pmin: float) -> Tuple[float, float]:
        for data, target in loader:
            return RiskEvaluator.compute_avg_losses(num_mc_samples=num_mc_samples,
                                                    model=model,
                                                    distribution=distribution,
                                                    data=data,
                                                    targets=target,
                                                    clamping=clamping,
                                                    pmin=pmin)
