import numpy as np
import torch
from torch.utils.data import dataloader
from tqdm import tqdm

from core.model.evaluation import RiskEvaluator
from core.model.probabilistic import AbstractPBPModel
from core.trainer.objective import AbstractObjective


class ModelEvaluator:
    @staticmethod
    def evaluate_stochastic(
        model: AbstractPBPModel,
        loader: dataloader,
        objective: AbstractObjective,
        samples: int,
        device: torch.device,
    ):
        # TODO: refactor
        model.eval()
        correct, cross_entropy, total = 0, 0.0, 0.0
        err_samples = np.zeros(samples)

        with torch.no_grad():
            for j in range(samples):
                for batch_id, (data, target) in enumerate(tqdm(loader, disable=True)):
                    # outputs = torch.zeros(len(target), pbobj.classes).to(self._device)
                    data = data.to(device)
                    target = target.to(device)
                    outputs = model(
                        data, sample=True, clamping=True, pmin=objective._pmin
                    )
                    cross_entropy += RiskEvaluator.compute_empirical_risk(
                        outputs, target.long(), True, objective._pmin).item()
                    pred = outputs.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                err_samples[j] = 1 - (correct / total)
        return cross_entropy / (batch_id + 1), np.mean(err_samples), np.std(err_samples)
