import torch
import torch.nn.functional as F
from torch import nn

from app.model import AbstractModel


class NN3Model(AbstractModel):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 dropout_probability: float,
                 device: torch.device):
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._dropout_probability = dropout_probability
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)
        self.d = nn.Dropout(dropout_probability)


    def forward(self, x):
        # forward pass for the network
        x = x.view(-1, self._input_dim)
        x = self.d(self.l1(x))
        x = F.relu(x)
        x = self.d(self.l2(x))
        x = F.relu(x)
        x = self.output_transform(self.l3(x), clamping=False, p_min=None)
        return x
