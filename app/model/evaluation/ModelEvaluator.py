import numpy as np
import torch
from torch.utils.data import dataloader
from tqdm import tqdm
from typing import Dict

from app.model.evaluation import RiskEvaluator
from app.model.probabilistic import AbstractPBPModel
from core.objective import AbstractObjective


class ModelEvaluator:
    @staticmethod
    def evaluate_stochastic(
        model: AbstractPBPModel,
        loader: dataloader,
        objective: AbstractObjective,
        samples: int,
        device: torch.device,
    ) -> Dict[str, float]:
        # TODO: refactor
        model.eval()
        correct, cross_entropy, total = 0, 0.0, 0.0
        err_samples = np.zeros(samples)

        with torch.no_grad():
            for j in range(samples):
                for batch_id, (data, target) in enumerate(tqdm(loader, disable=True)):
                    data = data.to(device)
                    target = target.to(device)
                    outputs = model(
                        data, sample=True, clamping=True, pmin=objective._pmin
                    )
                    cross_entropy += RiskEvaluator.compute_ce_empirical_loss(
                        outputs, target.long(), True, objective._pmin).item()
                    pred = outputs.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                err_samples[j] = 1 - (correct / total)
        # TODO: should remove np.std(err_samples)?
        return {'loss_ce': cross_entropy / (batch_id + 1),
                'loss_01': np.mean(err_samples)}

    @staticmethod
    def evaluate_risk(
            model: AbstractPBPModel,
            loader: dataloader,
            bound_loader: dataloader,
            objective: AbstractObjective,
            device: torch.device) -> Dict[str, float]:
        if len(bound_loader) > 1:
            raise NotImplementedError('Bound loader should have one batch')
        num_samples = len(loader) * loader.batch_size
        num_samples_bound = len(bound_loader) * bound_loader.batch_size
        model.eval()
        with torch.no_grad():
            for data, target in bound_loader:
                data, target = data.to(device), target.to(device)
                train_obj, kl, loss_ce_train, loss_01_train, risk_ce, risk_01 = objective.compute_risks(model,
                                                                                                       data,
                                                                                                       target,
                                                                                                       num_samples,
                                                                                                       num_samples_bound)
        return {'train_obj': train_obj.item(),
                'kl': kl.item(),
                'loss_ce_loader': loss_ce_train.item(),
                'loss_01_loader': loss_01_train.item(),
                'risk_ce': risk_ce.item(),
                'risk_01': risk_01.item()}
