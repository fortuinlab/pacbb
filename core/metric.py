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
    """
    Evaluate a set of metric functions on a test set with multiple Monte Carlo samples.

    This function uses `compute_losses` under the hood to compute each metric
    (e.g., NLL, 0-1 error) over `num_samples_metric` samples from the posterior.
    Optionally logs the results to Weights & Biases (wandb).

    Args:
        model (nn.Module): A probabilistic neural network model.
        metrics (Dict[str, Callable]): A dictionary mapping metric names 
            to metric functions (e.g., {"zero_one": zero_one_loss}).
        test_loader (DataLoader): DataLoader for the test/validation dataset.
        num_samples_metric (int): Number of Monte Carlo samples to draw 
            when evaluating each metric on the test set.
        device (torch.device): The device (CPU/GPU) to run computations on.
        pmin (float, optional): A lower bound for probabilities. If specified, 
            `bounded_call` is applied to model outputs.
        wandb_params (Dict, optional): Configuration for logging to wandb. 
            Expects keys:
            - "log_wandb": bool, whether to log or not
            - "name_wandb": str, prefix for logging metrics

    Returns:
        Dict[str, Tensor]: A dictionary mapping each metric name to its average value 
        across the entire test dataset and all Monte Carlo samples.
    """
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
