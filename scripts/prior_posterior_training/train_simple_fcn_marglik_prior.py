import torch
import logging
from torch.utils.tensorboard import SummaryWriter
from laplace import marglik_training

from core.bound import KLBound
from core.split_strategy import FaultySplitStrategy
from core.distribution.utils import from_copy, from_flat_rho
from core.distribution import GaussianVariable
from core.loss import compute_losses, scaled_nll_loss, zero_one_loss, nll_loss
from core.risk import evaluate
from core.training import train
from core.model import dnn_to_probnn
from core.objective import BBBObjective

from scripts.utils.dataset.loader import MNISTLoader
from scripts.utils.model import NNModel

logging.basicConfig(level=logging.INFO)

config = {
    'mcsamples': 1000,
    'pmin': 1e-5,
    'sigma': 0.01,
    'bound': {
        'delta': 0.025,
        'delta_test': 0.01,
    },
    'split_config': {
        'seed': 111,
        'dataset_loader_seed': 112,
        'batch_size': 250,
    },
    'dist_init': {
        'seed': 110,
    },
    'split_strategy': {
        'prior_type': 'learnt',
        'train_percent': 1.,
        'val_percent': 0.05,
        'prior_percent': .5,
        'self_certified': True,
    },
    'prior': {
        'training': {
            'kl_penalty': 0.001,
            'lr': 0.001,
            'momentum': 0.95,
            'epochs': 1,
            'seed': 1135,
            'hessian_structure': 'full'  # diag
        }
    },
    'posterior': {
        'training': {
            'kl_penalty': 1.0,
            'lr': 0.001,
            'momentum': 0.9,
            'epochs': 1,
            'seed': 1135,
        }
    }
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device ", device)
    # Losses
    losses = {'nll_loss': nll_loss, 'scaled_nll_loss': scaled_nll_loss, '01_loss': zero_one_loss}

    # Bound
    bound = KLBound(delta=config['bound']['delta'],
                    delta_test=config['bound']['delta_test'])

    # Data
    loader = MNISTLoader('./data/mnist')
    strategy = FaultySplitStrategy(prior_type=config['split_strategy']['prior_type'],
                                   train_percent=config['split_strategy']['train_percent'],
                                   val_percent=config['split_strategy']['val_percent'],
                                   prior_percent=config['split_strategy']['prior_percent'],
                                   self_certified=config['split_strategy']['self_certified'])
    strategy.split(loader, split_config=config['split_config'])

    # Model
    model = NNModel(28*28, 100, 10)

    # Training prior
    la, marglik_prior_model, _, _ = marglik_training(
        model=model,
        train_loader=strategy.prior_loader,
        likelihood='classification',
        hessian_structure=config['prior']['training']['hessian_structure'],
        n_epochs=config['prior']['training']['epochs'],
        optimizer_kwargs={'lr': config['prior']['training']['lr']},
        prior_structure='layerwise',
    )

    # Model
    model = NNModel(28*28, 100, 10)

    posterior_prior = from_flat_rho(model=marglik_prior_model,
                                    rho=torch.log(torch.exp(torch.sqrt(la.posterior_variance)) - 1),
                                    distribution=GaussianVariable,
                                    requires_grad=False)
    posterior = from_copy(dist=posterior_prior,
                          distribution=GaussianVariable,
                          requires_grad=True)
    dnn_to_probnn(model, posterior, posterior_prior)

    #  Train posterior
    train_params = {
        'lr': config['posterior']['training']['lr'],
        'momentum': config['posterior']['training']['momentum'],
        'epochs': config['posterior']['training']['epochs'],
        'seed': config['posterior']['training']['seed'],
        'num_samples': strategy.posterior_loader.batch_size * len(strategy.posterior_loader),
    }
    objective = BBBObjective(kl_penalty=config['posterior']['training']['kl_penalty'])
    train(model=model,
          posterior=posterior,
          prior=posterior_prior,
          objective=objective,
          train_loader=strategy.posterior_loader,
          val_loader=strategy.val_loader,
          parameters=train_params,
          device=device)

    # Compute average losses
    avg_losses = compute_losses(model=model,
                                bound_loader=strategy.bound_loader,
                                mc_samples=config['mcsamples'],
                                loss_func_list=list(losses.values()),
                                pmin=config['pmin'],
                                device=device)
    avg_losses = dict(zip(losses.keys(), avg_losses))
    print('avg_losses', avg_losses)

    # Evaluate bound
    for key, loss in avg_losses.items():
        result = evaluate(bound, loss,
                          posterior=posterior,
                          prior=posterior_prior,
                          mc_samples=config['mcsamples'],
                          bound_samples=strategy.bound_loader_1batch.batch_size * len(strategy.bound_loader_1batch))

        print(key, result)


if __name__ == '__main__':
    main()
