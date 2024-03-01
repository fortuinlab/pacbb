import numpy as np
import torch.nn.functional as F
from torch import Tensor


class RiskEvaluator:
    @staticmethod
    def compute_empirical_risk(
        outputs: Tensor, targets: Tensor, clamping: bool, pmin: float
    ) -> Tensor:
        empirical_risk = F.nll_loss(outputs, targets)
        if clamping:
            empirical_risk = empirical_risk / (np.log(1.0 / pmin))
        return empirical_risk

    @staticmethod
    def compute_01_empirical_risk(outputs: Tensor, targets: Tensor) -> Tensor:
        predictions = outputs.max(1, keepdim=True)[1]
        correct = predictions.eq(targets.view_as(predictions)).sum().item()
        total = targets.size(0)
        loss_01 = 1 - (correct / total)
        return Tensor([loss_01])
