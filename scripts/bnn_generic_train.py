from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn

from core.distribution.utils import from_copy, from_bnn
from core.distribution import GaussianVariable
from core.model import dnn_to_probnn

from scripts.utils.training import train_bnn

model = ...
const_bnn_prior_parameters = {
    "prior_mu": 0.0,
    "prior_sigma": config['sigma'],
    "posterior_mu_init": 0.0,
    "posterior_rho_init": -2.0,
    "type": "Reparameterization",
    "moped_enable": False,
    "moped_delta": 0.5,
}
dnn_to_bnn(model, const_bnn_prior_parameters)

train_params = {
    'lr': ...,
    'momentum': ...,
    'epochs': ...,
    'seed': ...,
    'num_samples': ...,
}

train_bnn(model=model,
          objective=objective,
          train_loader=strategy.prior_loader,
          val_loader=strategy.val_loader,
          parameters=train_params,
          device=device,
          wandb_params={'log_wandb': config["log_wandb"],
                        'name_wandb': 'Prior Train'})

d = from_bnn(model=model,
             distribution=GaussianVariable)

posterior_prior = from_copy(dist=d,
                            distribution=GaussianVariable,
                            requires_grad=False)
posterior = from_copy(dist=d,
                      distribution=GaussianVariable,
                      requires_grad=True)
dnn_to_probnn(model, weight_dist=posterior, prior_weight_dist=posterior_prior)

