from typing import List, Union, Any

import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter

# from app.model.probabilistic.distribution import AbstractVariable


class ModelUtils:
    @staticmethod
    def get_layers(model: nn.Module) -> List[nn.Module]:
        layers = []
        for layer in model.modules():
            if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                layers.append(layer)
        return layers

    @staticmethod
    def get_pbp_layers(model: nn.Module) -> List[nn.Module]:
        layers = []
        for layer in model.children():
            if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                layers.append(layer)
        return layers

    @staticmethod
    def sample_weight_call(model: nn.Module, distribution: List[Any], data: Tensor) -> Tensor:
        distribution_pairs = [(distribution[i], distribution[i+1]) for i in range(0, len(distribution), 2)]
        for layer, (weight_dist, bias_dist) in zip(ModelUtils.get_layers(model), distribution_pairs):
            layer.weight = Parameter(weight_dist.sample())
            layer.bias = Parameter(bias_dist.sample())
        return model(data)
