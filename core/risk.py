import logging
import wandb
from typing import Dict, Callable
import torch
from torch import Tensor, nn

from core.distribution.utils import DistributionT
from core.bound import AbstractBound
from core.distribution.utils import compute_kl
from core.loss import compute_losses


def certify_risk(model: nn.Module,
                 bounds: Dict[str, AbstractBound],
                 losses: Dict[str, Callable],
                 posterior: DistributionT,
                 prior: DistributionT,
                 bound_loader: torch.utils.data.dataloader.DataLoader,
                 num_samples_loss: int,
                 device: torch.device,
                 pmin: float = 1e-5,
                 wandb_params: Dict = None,
                 ) -> Dict[str, Dict[str, Dict[str, Tensor]]]:
    """
    Certify (evaluate) the generalization risk of a probabilistic neural network
    using one or more PAC-Bayes bounds on a given dataset.

    Steps:
      1) Compute average losses (e.g., NLL, 0-1 error) via multiple Monte Carlo samples
         from the posterior (`compute_losses`).
      2) Calculate the KL divergence between the posterior and prior distributions.
      3) For each bound in `bounds`, calculate a PAC-Bayes risk bound for each loss in `losses`.
      4) Optionally log intermediate results (loss, risk) to Weights & Biases (wandb).

    Args:
        model (nn.Module): The probabilistic neural network used for risk evaluation.
        bounds (Dict[str, AbstractBound]): A mapping from bound names to bound objects 
            that implement a PAC-Bayes bound (`AbstractBound`).
        losses (Dict[str, Callable]): A mapping from loss names to loss functions 
            (e.g., {"nll": nll_loss, "01": zero_one_loss}).
        posterior (DistributionT): Posterior distribution of the model parameters.
        prior (DistributionT): Prior distribution of the model parameters.
        bound_loader (DataLoader): DataLoader for the dataset on which bounds and losses are computed.
        num_samples_loss (int): Number of Monte Carlo samples to draw from the posterior
            for estimating the average losses.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        pmin (float, optional): A minimum probability bound for clamping model outputs in log space.
            Defaults to 1e-5.
        wandb_params (Dict, optional): Configuration for Weights & Biases logging. Expects keys:
            - "log_wandb": bool, whether to log
            - "name_wandb": str, prefix for metric names

    Returns:
        Dict[str, Dict[str, Dict[str, Tensor]]]: A nested dictionary of the form:
            {
              bound_name: {
                loss_name: {
                  'risk': risk_value,
                  'loss': avg_loss_value
                }
              }
            }
        where `risk_value` is the computed bound on the risk, and `avg_loss_value` is the
        empirical loss estimate for that loss and bound.
    """
    avg_losses = compute_losses(model=model,
                                bound_loader=bound_loader,
                                mc_samples=num_samples_loss,
                                loss_func_list=list(losses.values()),
                                pmin=pmin,
                                device=device)
    avg_losses = dict(zip(losses.keys(), avg_losses))
    logging.info('Average losses:')
    logging.info(avg_losses)

    # Evaluate bound
    kl = compute_kl(dist1=posterior, dist2=prior)
    num_samples_bound = len(bound_loader.sampler)

    result = {}
    for bound_name, bound in bounds.items():
        logging.info(f'Bound name: {bound_name}')
        result[bound_name] = {}
        for loss_name, avg_loss in avg_losses.items():
            risk, loss = bound.calculate(avg_loss=avg_loss,
                                         kl=kl,
                                         num_samples_bound=num_samples_bound,
                                         num_samples_loss=num_samples_loss)
            result[bound_name][loss_name] = {'risk': risk, 'loss': loss}
            logging.info(f'Loss name: {loss_name}, '
                         f'Risk: {risk.item():.5f}, '
                         f'Loss: {loss.item():.5f}, '
                         f'KL per sample bound: {kl / num_samples_bound:.5f}')
            if wandb_params is not None and wandb_params['log_wandb']:
                wandb.log({f'{wandb_params["name_wandb"]}/{bound_name}/{loss_name}_loss': loss.item(),
                           f'{wandb_params["name_wandb"]}/{bound_name}/{loss_name}_risk': risk.item()})
    if wandb_params is not None and wandb_params['log_wandb']:
        wandb.log({f'{wandb_params["name_wandb"]}/KL-n/': kl / num_samples_bound})

    return result
