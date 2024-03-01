from core.pipeline.training import (AbstractTrainingPipeline,
                                    PBPTrainingPipeline)
from core.utils import AbstractFactory


class TrainingPipelineFactory(AbstractFactory):

    def __init__(self) -> None:
        super().__init__()
        self.register_creator("pbp", PBPTrainingPipeline)

    def create(self, training_pipeline_name: str) -> AbstractTrainingPipeline:
        creator = self._creators.get(training_pipeline_name)
        if not creator:
            raise ValueError(f"Invalid training pipeline: {training_pipeline_name}")
        return creator()
