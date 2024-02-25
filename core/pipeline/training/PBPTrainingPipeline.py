from typing import Dict

from core.dataset import DatasetHandler
from core.pipeline.training import AbstractTrainingPipeline
from core.trainer import TrainerFactory


class PBPTrainingPipeline(AbstractTrainingPipeline):
    def __init__(self):
        super().__init__()

    def train(
        self,
        training_pipeline_config: Dict,
        dataset_config: Dict,
        split_strategy_config: Dict,
    ) -> None:
        self._dataset_handler = DatasetHandler(dataset_config, split_strategy_config)
        self._dataset_handler.load_and_split_dataset()

        prior_trainer = TrainerFactory().create(
            training_pipeline_config["prior"]["trainer_name"]
        )
        prior_model = PBPModel()
        prior_trainer.train(
            model=prior_model, training_config=training_pipeline_config["prior"]
        )

        posterior_trainer = TrainerFactory().create(
            training_pipeline_config["posterior"]["trainer_name"]
        )
        posterior_model = PBPModel(prior_model)
        posterior_trainer.train(
            model=posterior_model, training_config=training_pipeline_config["posterior"]
        )

        self._prior_model = prior_model
        self._posterior_model = posterior_model
