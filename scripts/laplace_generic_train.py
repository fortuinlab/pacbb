import torch

from core.distribution.utils import from_copy, from_flat_rho
from core.distribution import GaussianVariable
from core.model import dnn_to_probnn

from scripts.utils.training import train_laplace

model = ...

train_params = {
    'lr': config['prior']['training']['lr'],
    'momentum': config['prior']['training']['momentum'],
    'epochs': config['prior']['training']['epochs'],
    'seed': config['prior']['training']['seed'],
    'num_samples': strategy.prior_loader.batch_size * len(strategy.prior_loader),
    'train_samples': config['prior']['training']['train_samples'],
    'sigma': config['sigma'],
}

la = train_laplace(model=model,
                   train_loader=strategy.prior_loader,
                   val_loader=strategy.val_loader,
                   parameters=train_params,
                   device=device,
                   wandb_params={'log_wandb': config["log_wandb"],
                                 'name_wandb': 'Prior Train'})

posterior_prior = from_flat_rho(model=model,
                                rho=torch.log(torch.exp(torch.sqrt(la.posterior_variance)) - 1),
                                distribution=GaussianVariable,
                                requires_grad=False)
posterior = from_copy(dist=posterior_prior,
                      distribution=GaussianVariable,
                      requires_grad=True)
dnn_to_probnn(model, weight_dist=posterior, prior_weight_dist=posterior_prior)
model.to(device)
