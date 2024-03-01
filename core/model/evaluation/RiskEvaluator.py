from torch import Tensor
import numpy as np
import torch.nn.functional as F


class RiskEvaluator:
    @staticmethod
    def compute_empirical_risk(outputs: Tensor, targets: Tensor, pmin: float, bounded=True) -> Tensor:
        empirical_risk = F.nll_loss(outputs, targets)
        if bounded:
            empirical_risk = empirical_risk / (np.log(1. / pmin))
        return empirical_risk

    @staticmethod
    def compute_01_empirical_risk(outputs: Tensor, targets: Tensor) -> Tensor:
        predictions = outputs.max(1, keepdim=True)[1]
        correct = predictions.eq(targets.view_as(predictions)).sum().item()
        total = targets.size(0)
        loss_01 = 1 - (correct / total)
        return Tensor([loss_01])
