from typing import Callable
import torchmetrics

from scripts.utils.factory import AbstractFactory


class MetricFactory(AbstractFactory[Callable]):
    def __init__(self) -> None:
        super().__init__()
        self.register_creator("accuracy_micro_metric",
                              lambda o, t, pmin: torchmetrics.functional.accuracy(o, t, task='multiclass',
                                                                                  num_classes=10, average='micro'))
        self.register_creator("f1_micro_metric",
                              lambda o, t, pmin: torchmetrics.functional.f1_score(o, t, task='multiclass',
                                                                                  num_classes=10, average='micro'))
        self.register_creator("accuracy_macro_metric",
                              lambda o, t, pmin: torchmetrics.functional.accuracy(o, t, task='multiclass',
                                                                                  num_classes=10, average='macro'))
        self.register_creator("f1_macro_metric",
                              lambda o, t, pmin: torchmetrics.functional.f1_score(o, t, task='multiclass',
                                                                                  num_classes=10, average='macro'))

    def create(self, name: str, *args, **kwargs) -> Callable:
        creator = self._creators.get(name)
        if not creator:
            raise ValueError(f"Invalid creator: {name}")
        return creator
