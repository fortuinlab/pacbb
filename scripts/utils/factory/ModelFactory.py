from torch import nn
from scripts.utils.model import NNModel, ConvNNModel, ConvNN15Model, ResNet, GoogLeNet

from scripts.utils.factory import AbstractFactory


class ModelFactory(AbstractFactory[nn.Module]):
    def __init__(self) -> None:
        super().__init__()
        self.register_creator("nn", NNModel)
        self.register_creator("conv", ConvNNModel)
        self.register_creator("conv15", ConvNN15Model)
        self.register_creator("resnet", ResNet)
        self.register_creator("googlenet", GoogLeNet)
