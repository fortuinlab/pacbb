from abc import ABC, abstractmethod
import torch


class KLDivergenceInterface(ABC):
    @abstractmethod
    def compute_kl(self, *args, **kwargs) -> torch.Tensor:
        pass
