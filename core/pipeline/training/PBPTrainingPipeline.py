from typing import Dict

import torch

from core.dataset import DatasetHandler
from core.model.probabilistic import PBP3Model
from core.pipeline.training import AbstractTrainingPipeline
from core.trainer import TrainerFactory
from core.trainer import TorchOptimizerFactory
from core.trainer.objective import ObjectiveFactory


class PBPTrainingPipeline(AbstractTrainingPipeline):
    def __init__(self):
        super().__init__()

    def train(
        self,
        training_pipeline_config: Dict,
        dataset_config: Dict,
        split_strategy_config: Dict,
        device: torch.device,
    ) -> None:
        model_config = training_pipeline_config["model"]
        prior_config = training_pipeline_config["prior"]
        posterior_config = training_pipeline_config["posterior"]

        # Prepare data
        self._dataset_handler = DatasetHandler(dataset_config, split_strategy_config)
        self._dataset_handler.load_and_split_dataset()

        prior_loader = self._dataset_handler.split_strategy.prior_loader

        # Select model
        # TODO: implement model factory
        prior_model = PBP3Model(
            model_weight_distribution=model_config["model_weight_distribution"],
            sigma=model_config["sigma"],
            weight_initialization_method=model_config["weight_initialization_method"],
            input_dim=model_config["input_dim"],
            output_dim=model_config["output_dim"],
            hidden_dim=model_config["hidden_dim"],
            device=device,
        )

        # Prior training

        # Select optimizer
        optimizer_config = prior_config['objective'].pop('objective_name', None)
        optimizer_config['params'] = prior_model.parameters()
        prior_optimizer = TorchOptimizerFactory().create(prior_config['optimizer']['optimizer_name'],
                                                         **optimizer_config)

        # Select objective
        objective_config = prior_config['optimizer'].pop('optimizer_name', None)
        objective_config['num_classes'] = model_config['output_dim']
        objective_config['device'] = device
        prior_objective = ObjectiveFactory().create(prior_config['objective']['objective_name'],
                                                    **objective_config)

        # Select trainer
        prior_trainer = TrainerFactory().create(prior_config["trainer_name"], device)

        # Train model
        training_config = {
            'epochs': prior_config['epochs'],
            'disable_tqdm': training_pipeline_config['disable_tqdm'],
            'train_loader': prior_loader,
        }
        prior_trainer.train(
            model=prior_model,
            optimizer=prior_optimizer,
            objective=prior_objective,
            training_config=training_config
        )

        # Posterior training

        posterior_trainer = TrainerFactory().create(
            training_pipeline_config["posterior"]["trainer_name"]
        )
        posterior_model = PBP3Model(
            model_weight_distribution=model_config["model_weight_distribution"],
            sigma=model_config["sigma"],
            weight_initialization_method=model_config["weight_initialization_method"],
            input_dim=model_config["input_dim"],
            output_dim=model_config["output_dim"],
            hidden_dim=model_config["hidden_dim"],
            device=device,
        )
        posterior_model.set_weights_from_model(prior_model)
        posterior_trainer.train(
            model=posterior_model, training_config=training_pipeline_config["posterior"]
        )

        self._prior_model = prior_model
        self._posterior_model = posterior_model
