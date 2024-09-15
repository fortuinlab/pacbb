import wandb
import torch
import logging

from core.split_strategy import PBPSplitStrategy
from core.distribution.utils import from_copy, from_zeros, from_random
from core.distribution import GaussianVariable
from core.training import train
from core.model import dnn_to_probnn, update_dist
from core.risk import certify_risk
from core.metric import evaluate_metrics

from scripts.utils.factory import (LossFactory,
                                   MetricFactory,
                                   BoundFactory,
                                   DataLoaderFactory,
                                   ModelFactory,
                                   ObjectiveFactory)

logging.basicConfig(level=logging.INFO)

config = {
    'log_wandb': True,
    'mcsamples': 1000,
    'pmin': 1e-5,
    'sigma': 0.1,
    'factory':
        {
            'losses': ['nll_loss', 'scaled_nll_loss', '01_loss'],
            'metrics': ['accuracy_micro_metric', 'accuracy_macro_metric', 'f1_micro_metric', 'f1_macro_metric'],
            'bounds': ['kl', 'mcallister'],
            'data_loader': {'name': 'cifar10',
                            'params': {'dataset_path': './data/cifar10'}
                            },
            'model': {'name': 'conv15',
                      'params': {'in_channels': 3, 'dataset': 'cifar10'}
                      },
            'prior_objective': {'name': 'fclassic',
                                'params': {'kl_penalty': 0.01,
                                           'delta': 0.025,
                                           }
                                },
            'posterior_objective': {'name': 'fclassic',
                                    'params': {'kl_penalty': 1.0,
                                               'delta': 0.025
                                               }
                                    },
         },
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
        'train_percent': 1.0,
        'val_percent': 0.0,
        'prior_percent': .7,
        'self_certified': True,
    },
    'prior': {
        'training': {
            'lr': 0.01,
            'momentum': 0.95,
            'epochs': 100,
            'seed': 1135,
        }
    },
    'posterior': {
        'training': {
            'lr': 0.001,
            'momentum': 0.9,
            'epochs': 1,
            'seed': 1135,
        }
    }
}


def main():
    if config['log_wandb']:
        wandb.init(project='pbb-framework', config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device ", device)
    # Losses
    logging.info(f'Selected losses: {config["factory"]["losses"]}')
    loss_factory = LossFactory()
    losses = {loss_name: loss_factory.create(loss_name) for loss_name in config["factory"]["losses"]}

    # Metrics
    logging.info(f'Select metrics: {config["factory"]["metrics"]}')
    metric_factory = MetricFactory()
    metrics = {metric_name: metric_factory.create(metric_name) for metric_name in config["factory"]["metrics"]}

    # Bound
    logging.info(f'Selected bounds: {config["factory"]["bounds"]}')
    bound_factory = BoundFactory()
    bounds = {bound_name: bound_factory.create(bound_name,
                                               bound_delta=config['bound']['delta'],
                                               loss_delta=config['bound']['delta_test'])
              for bound_name in config["factory"]["bounds"]}

    # Data
    logging.info(f'Selected data loader: {config["factory"]["data_loader"]}')
    data_loader_factory = DataLoaderFactory()
    loader = data_loader_factory.create(config["factory"]["data_loader"]["name"],
                                        **config["factory"]["data_loader"]["params"])

    strategy = PBPSplitStrategy(prior_type=config['split_strategy']['prior_type'],
                                train_percent=config['split_strategy']['train_percent'],
                                val_percent=config['split_strategy']['val_percent'],
                                prior_percent=config['split_strategy']['prior_percent'],
                                self_certified=config['split_strategy']['self_certified'])
    strategy.split(loader, split_config=config['split_config'])

    # Model
    logging.info(f'Select model: {config["factory"]["model"]["name"]}')
    model_factory = ModelFactory()
    model = model_factory.create(config["factory"]["model"]["name"], **config["factory"]["model"]["params"])

    torch.manual_seed(config['dist_init']['seed'])
    prior_prior = from_zeros(model=model,
                             rho=torch.log(torch.exp(torch.Tensor([config['sigma']])) - 1),
                             distribution=GaussianVariable,
                             requires_grad=False)
    prior = from_random(model=model,
                        rho=torch.log(torch.exp(torch.Tensor([config['sigma']])) - 1),
                        distribution=GaussianVariable,
                        requires_grad=True)
    dnn_to_probnn(model, prior, prior_prior)
    model.to(device)

    # Training prior
    train_params = {
        'lr': config['prior']['training']['lr'],
        'momentum': config['prior']['training']['momentum'],
        'epochs': config['prior']['training']['epochs'],
        'seed': config['prior']['training']['seed'],
        'num_samples': strategy.prior_loader.batch_size * len(strategy.prior_loader),
    }
    logging.info(f'Select objective: {config["factory"]["prior_objective"]["name"]}')
    objective_factory = ObjectiveFactory()
    objective = objective_factory.create(config["factory"]["prior_objective"]["name"],
                                         **config["factory"]["prior_objective"]["params"])

    train(model=model,
          posterior=prior,
          prior=prior_prior,
          objective=objective,
          train_loader=strategy.prior_loader,
          val_loader=strategy.val_loader,
          parameters=train_params,
          device=device,
          wandb_params={'log_wandb': config["log_wandb"],
                        'name_wandb': 'Prior Train'})

    # if strategy.test_loader is not None:
    #     _  = evaluate_metrics(model=model,
    #                           metrics=metrics,
    #                           test_loader=strategy.test_loader,
    #                           num_samples_metric=config["mcsamples"],
    #                           device=device,
    #                           pmin=config["pmin"],
    #                           wandb_params={'log_wandb': config["log_wandb"],
    #                                         'name_wandb': 'Prior Evaluation'})

    _ = certify_risk(model=model,
                     bounds=bounds,
                     losses=losses,
                     posterior=prior,
                     prior=prior_prior,
                     bound_loader=strategy.bound_loader,
                     num_samples_loss=config["mcsamples"],
                     device=device,
                     pmin=config["pmin"],
                     wandb_params={'log_wandb': config["log_wandb"],
                                   'name_wandb': 'Prior Bound'})

    posterior_prior = from_copy(dist=prior,
                                distribution=GaussianVariable,
                                requires_grad=False)
    posterior = from_copy(dist=prior,
                          distribution=GaussianVariable,
                          requires_grad=True)
    update_dist(model, weight_dist=posterior, prior_weight_dist=posterior_prior)
    model.to(device)

    #  Train posterior
    train_params = {
        'lr': config['posterior']['training']['lr'],
        'momentum': config['posterior']['training']['momentum'],
        'epochs': config['posterior']['training']['epochs'],
        'seed': config['posterior']['training']['seed'],
        'num_samples': strategy.posterior_loader.batch_size * len(strategy.posterior_loader),
    }

    logging.info(f'Select objective: {config["factory"]["posterior_objective"]["name"]}')
    objective = objective_factory.create(config["factory"]["posterior_objective"]["name"],
                                         **config["factory"]["posterior_objective"]["params"])

    train(model=model,
          posterior=posterior,
          prior=posterior_prior,
          objective=objective,
          train_loader=strategy.posterior_loader,
          val_loader=strategy.val_loader,
          parameters=train_params,
          device=device,
          wandb_params={'log_wandb': config["log_wandb"],
                        'name_wandb': 'Posterior Train'})

    # if strategy.test_loader is not None:
    #     _ = evaluate_metrics(model=model,
    #                          metrics=metrics,
    #                          test_loader=strategy.test_loader,
    #                          num_samples_metric=config["mcsamples"],
    #                          device=device,
    #                          pmin=config["pmin"],
    #                          wandb_params={'log_wandb': config["log_wandb"],
    #                                        'name_wandb': 'Posterior Evaluation'})

    _ = certify_risk(model=model,
                     bounds=bounds,
                     losses=losses,
                     posterior=posterior,
                     prior=posterior_prior,
                     bound_loader=strategy.bound_loader,
                     num_samples_loss=config["mcsamples"],
                     device=device,
                     pmin=config["pmin"],
                     wandb_params={'log_wandb': config["log_wandb"],
                                   'name_wandb': 'Posterior Bound'})


if __name__ == '__main__':
    main()
