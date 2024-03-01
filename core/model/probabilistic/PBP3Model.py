import torch
import torch.nn.functional as F
from torch import Tensor

from core.model.probabilistic import AbstractPBPModel
from core.model.probabilistic.layer import ProbabilisticLinearLayer


class PBP3Model(AbstractPBPModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        model_weight_distribution: str,
        sigma: float,
        weight_initialization_method: str,
        device: torch.device,
    ):
        super().__init__(
            model_weight_distribution,
            sigma,
            weight_initialization_method,
            input_dim,
            output_dim,
            hidden_dim,
            device,
        )
        self.l1 = ProbabilisticLinearLayer(
            self._input_dim,
            self._hidden_dim,
            self._model_weight_distribution,
            self._sigma,
            self._weight_initialization_method,
            self._device,
        )
        self.l2 = ProbabilisticLinearLayer(
            self._hidden_dim,
            self._hidden_dim,
            self._model_weight_distribution,
            self._sigma,
            self._weight_initialization_method,
            self._device,
        )
        self.l3 = ProbabilisticLinearLayer(
            self._hidden_dim,
            self._output_dim,
            self._model_weight_distribution,
            self._sigma,
            self._weight_initialization_method,
            self._device,
        )

    def forward(self, x: Tensor, sample: bool, clamping: bool, pmin: float) -> Tensor:
        # TODO: move default values to settings
        x = x.view(-1, self._input_dim)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = self.l3(x, sample)
        x = self.output_transform(x, clamping, pmin)
        return x

    def compute_kl(self, recompute: bool = True) -> Tensor:
        if recompute:
            return self.l1.compute_kl() + self.l2.compute_kl() + self.l3.compute_kl()
        else:
            return self.l1.kl_div + self.l2.kl_div + self.l3.kl_div

    def set_weights_from_model(self, model: "PBP3Model") -> None:
        self.l1.set_weights_from_layer(model.l1)
        self.l2.set_weights_from_layer(model.l2)
        self.l3.set_weights_from_layer(model.l3)
