import json
from typing import Dict

import torch

from core.dataset import DatasetHandler
from core.model.probabilistic import PBP3Model
from core.pipeline.training import AbstractTrainingPipeline
from core.trainer import TorchOptimizerFactory, TrainerFactory
from core.trainer.objective import ObjectiveFactory
from core.model.evaluation import ModelEvaluator
from core.trainer.callback import StochasticCallback, BoundCallback
from core.utils import logger


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
        logger.info("Prepare data")
        self._dataset_handler = DatasetHandler(dataset_config, split_strategy_config)
        self._dataset_handler.load_and_split_dataset()

        prior_loader = self._dataset_handler.split_strategy.prior_loader
        val_loader = self._dataset_handler.split_strategy.val_loader
        posterior_loader = self._dataset_handler.split_strategy.posterior_loader
        bound_loader_1batch = self._dataset_handler.split_strategy.bound_loader_1batch

        # Prior training
        logger.info("Train prior")

        # Select model
        logger.info("Select model")
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

        # Select optimizer
        optimizer_config = prior_config["optimizer"].copy()
        optimizer_config.pop("optimizer_name", None)
        optimizer_config["params"] = prior_model.parameters()
        prior_optimizer = TorchOptimizerFactory().create(
            prior_config["optimizer"]["optimizer_name"], **optimizer_config
        )

        # Select objective
        objective_config = prior_config["objective"].copy()
        objective_config.pop("objective_name", None)
        objective_config["num_classes"] = model_config["output_dim"]
        objective_config["device"] = device
        objective_config["num_mc_samples"] = training_pipeline_config["num_mc_samples"]
        objective_config["delta"] = training_pipeline_config["delta"]
        objective_config["delta_test"] = training_pipeline_config["delta_test"]
        prior_objective = ObjectiveFactory().create(
            prior_config["objective"]["objective_name"], **objective_config
        )

        # Select trainer
        prior_trainer = TrainerFactory().create(prior_config["trainer_name"], StochasticCallback(device), device)

        # Train model
        training_config = {
            "epochs": prior_config["epochs"],
            "disable_tqdm": training_pipeline_config["disable_tqdm"],
            "train_loader": prior_loader,
            "val_loader": val_loader,
            "num_samples": len(prior_loader) * prior_loader.batch_size,
        }
        prior_model = prior_trainer.train(
            model=prior_model,
            optimizer=prior_optimizer,
            objective=prior_objective,
            training_config=training_config,
        )

        # Evaluate model
        logger.info("Evaluate")
        model_evaluation_dict = ModelEvaluator.evaluate_risk(prior_model,
                                                             prior_loader,
                                                             bound_loader_1batch,
                                                             prior_objective,
                                                             device)
        logger.info(json.dumps(model_evaluation_dict))

        # Posterior training
        logger.info("Train posterior")

        # Select model
        logger.info("Select model")
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

        # Select optimizer
        optimizer_config = posterior_config["optimizer"].copy()
        optimizer_config.pop("optimizer_name", None)
        optimizer_config["params"] = posterior_model.parameters()
        posterior_optimizer = TorchOptimizerFactory().create(
            posterior_config["optimizer"]["optimizer_name"], **optimizer_config
        )

        # Select objective
        objective_config = posterior_config["objective"].copy()
        objective_config.pop("objective_name", None)
        objective_config["num_classes"] = model_config["output_dim"]
        objective_config["device"] = device
        objective_config["num_mc_samples"] = training_pipeline_config["num_mc_samples"]
        objective_config["delta"] = training_pipeline_config["delta"]
        objective_config["delta_test"] = training_pipeline_config["delta_test"]
        posterior_objective = ObjectiveFactory().create(
            posterior_config["objective"]["objective_name"], **objective_config
        )

        # TODO: move freq_test to prams
        freq_test = 3
        # Select trainer
        posterior_trainer = TrainerFactory().create(posterior_config["trainer_name"], BoundCallback(freq_test, device), device)

        # Train model
        training_config = {
            "epochs": posterior_config["epochs"],
            "disable_tqdm": training_pipeline_config["disable_tqdm"],
            "train_loader": posterior_loader,
            "val_loader": bound_loader_1batch,
            "num_samples": len(posterior_loader) * posterior_loader.batch_size,
        }
        posterior_model = posterior_trainer.train(
            model=posterior_model,
            optimizer=posterior_optimizer,
            objective=posterior_objective,
            training_config=training_config,
        )

        # Evaluate model
        logger.info("Evaluate")
        model_evaluation_dict = ModelEvaluator.evaluate_risk(posterior_model,
                                                             posterior_loader,
                                                             bound_loader_1batch,
                                                             posterior_objective,
                                                             device)
        logger.info(json.dumps(model_evaluation_dict))
        
        # Save models
        self._prior_model = prior_model
        self._posterior_model = posterior_model
