import logging
import wandb
from typing import Dict, Callable
import torch
from torch import Tensor, nn

from core.loss import compute_losses


def evaluate_metrics(model: nn.Module,
                     metrics: Dict[str, Callable],
                     test_loader: torch.utils.data.dataloader.DataLoader,
                     num_samples_metric: int,
                     device: torch.device,
                     pmin: float = 1e-5,
                     wandb_params: Dict = None,
                     ) -> Dict[str, Tensor]:
    # Compute average metrics
    avg_metrics = compute_losses(model=model,
                                 bound_loader=test_loader,
                                 mc_samples=num_samples_metric,
                                 loss_func_list=list(metrics.values()),
                                 pmin=pmin,
                                 device=device)
    avg_metrics = dict(zip(metrics.keys(), avg_metrics))
    logging.info('Average metrics:')
    logging.info(avg_metrics)
    if wandb_params is not None and wandb_params['log_wandb']:
        for name, metric in avg_metrics.items():
            wandb.log({f'{wandb_params["name_wandb"]}/{name}': metric.item()})
    return avg_metrics
