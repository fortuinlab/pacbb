import torch.nn as nn

from core.model.probabilistic import AbstractProbabilisticModel


class SequentialProbabilisticModel(AbstractProbabilisticModel):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()
        raise NotImplementedError()
