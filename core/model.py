from collections.abc import Callable, Iterator

import numpy as np
import torch
from torch import Tensor, nn

from core.distribution.utils import DistributionT
from core.layer import LAYER_MAPPING, AbstractProbLayer
from core.layer.utils import LayerNameT, get_torch_layers


def bounded_call(model: nn.Module, data: Tensor, pmin: float) -> Tensor:
    """
    Forward data through the model and clamp the output to a minimum log-probability.

    This is typically used to avoid numerical instability in PAC-Bayes experiments
    when dealing with small probabilities. The output is clamped at log(pmin) to
    ensure log-probabilities do not fall below this threshold.

    Args:
        model (nn.Module): The (probabilistic) neural network model.
        data (Tensor): Input data of shape (batch_size, ...).
        pmin (float): A lower bound for probabilities. Outputs are clamped at log(pmin).

    Returns:
        Tensor: Model outputs with each element >= log(pmin).
    """
    return torch.clamp(model(data), min=np.log(pmin))


def dnn_to_probnn(
    model: nn.Module,
    weight_dist: DistributionT,
    prior_weight_dist: DistributionT,
    get_layers_func: Callable[
        [nn.Module], Iterator[tuple[LayerNameT, nn.Module]]
    ] = get_torch_layers,
):
    """
    Convert a deterministic PyTorch model into a probabilistic neural network (ProbNN)
    by attaching weight/bias distributions to its layers.

    This function iterates through each layer in the model (using `get_layers_func`),
    and if the layer type is supported, it:
      - Registers prior and posterior distributions for weights and biases.
      - Marks the layer as probabilistic (so that it samples weights/biases in forward calls).
      - Replaces the layer class with its probabilistic counterpart from `LAYER_MAPPING`.

    Args:
        model (nn.Module): A deterministic PyTorch model (e.g., a CNN).
        weight_dist (DistributionT): A dictionary containing posterior distributions
            for weights and biases, keyed by layer name.
        prior_weight_dist (DistributionT): A dictionary containing prior distributions
            for weights and biases, keyed by layer name.
        get_layers_func (Callable): A function that returns an iterator of (layer_name, layer_module)
            pairs. Defaults to `get_torch_layers`.

    Returns:
        None: The function modifies `model` in place, converting certain layers
        to their probabilistic equivalents.
    """
    for name, layer in get_layers_func(model):
        layer_type = type(layer)
        if layer_type in LAYER_MAPPING:
            layer.register_module(
                "_prior_weight_dist", prior_weight_dist[name]["weight"]
            )
            layer.register_module("_prior_bias_dist", prior_weight_dist[name]["bias"])
            layer.register_module("_weight_dist", weight_dist[name]["weight"])
            layer.register_module("_bias_dist", weight_dist[name]["bias"])
            layer.__setattr__("probabilistic_mode", True)
            layer.__class__ = LAYER_MAPPING[layer_type]
    model.probabilistic = AbstractProbLayer.probabilistic.__get__(model, nn.Module)


def update_dist(
    model: nn.Module,
    weight_dist: DistributionT = None,
    prior_weight_dist: DistributionT = None,
    get_layers_func: Callable[
        [nn.Module], Iterator[tuple[LayerNameT, nn.Module]]
    ] = get_torch_layers,
):
    """
    Update the weight/bias distributions of an already converted probabilistic model.

    This is useful when you want to load a different set of posterior or prior
    distributions into the same network structure, without re-running the entire
    `dnn_to_probnn` procedure.

    Args:
        model (nn.Module): The probabilistic neural network model (already converted).
        weight_dist (DistributionT, optional): New posterior distributions keyed by layer name.
            If provided, each layer's '_weight_dist' and '_bias_dist' are updated.
        prior_weight_dist (DistributionT, optional): New prior distributions keyed by layer name.
            If provided, each layer's '_prior_weight_dist' and '_prior_bias_dist' are updated.
        get_layers_func (Callable): Function that returns an iterator of (layer_name, layer_module).
            Defaults to `get_torch_layers`.

    Returns:
        None: The distributions in the model are updated in place.
    """
    if weight_dist is not None:
        for (_name, distribution), (_, layer) in zip(
            weight_dist.items(), get_layers_func(model), strict=False
        ):
            layer_type = type(layer)
            if layer_type in LAYER_MAPPING.values():
                layer.__setattr__("_weight_dist", distribution["weight"])
                layer.__setattr__("_bias_dist", distribution["bias"])

    if prior_weight_dist is not None:
        for (_name, distribution), (_, layer) in zip(
            prior_weight_dist.items(), get_layers_func(model), strict=False
        ):
            layer_type = type(layer)
            if layer_type in LAYER_MAPPING.values():
                layer.__setattr__("_prior_weight_dist", distribution["weight"])
                layer.__setattr__("_prior_bias_dist", distribution["bias"])
