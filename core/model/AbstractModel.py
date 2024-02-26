from abc import ABC, abstractmethod
from torch import nn


class AbstractModel(nn.Module, ABC):
    pass
