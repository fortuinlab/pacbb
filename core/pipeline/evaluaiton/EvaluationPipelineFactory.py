from core.pipeline.evaluaiton import (AbstractEvaluationPipeline,
                                      PBPEvaluationPipeline)
from core.utils import AbstractFactory


class EvaluationPipelineFactory(AbstractFactory):

    def __init__(self) -> None:
        super().__init__()
        self.register_creator("pbp", PBPEvaluationPipeline)

    def create(self, evaluation_pipeline_name: str) -> AbstractEvaluationPipeline:
        creator = self._creators.get(evaluation_pipeline_name)
        if not creator:
            raise ValueError(f"Invalid evaluation pipeline: {evaluation_pipeline_name}")
        return creator(evaluation_pipeline_name)
