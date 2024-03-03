from abc import ABC, abstractmethod

from core.pipeline.evaluaiton import AbstractEvaluationPipeline


class PBPEvaluationPipeline(AbstractEvaluationPipeline, ABC):
    pass

    def evaluate(self, ) -> None:
        pass
