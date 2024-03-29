from abc import ABC

from torch import nn

from core.distribution import AbstractVariable


class AbstractProbLayer(nn.Module, ABC):
    probabilistic_mode: bool
    _weight_dist: AbstractVariable
    _bias_dist: AbstractVariable

    def probabilistic(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("probabilistic mode is expected to be boolean")
        if isinstance(self, AbstractProbLayer):
            self.probabilistic_mode = mode
        for module in self.children():
            if hasattr(module, 'probabilistic_mode') and hasattr(module, 'probabilistic'):
                module.probabilistic(mode)
