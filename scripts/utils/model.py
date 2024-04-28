from torch import nn
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torchvision import models


class NNModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super(NNModel, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self._input_dim)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.log_softmax(x, dim=1)
        return x


class ConvNNModel(nn.Module):
    def __init__(self, in_channels: int = 1, dataset='mnist'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        if dataset == 'mnist':
            self.fc1 = nn.Linear(9216, 128)
        elif dataset == 'cifar10':
            self.fc1 = nn.Linear(12544, 128)
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class GoogLeNet(pl.LightningModule):
    def __init__(self, num_classes=10, num_channels: int = 1):
        super().__init__()
        self.model = models.googlenet(weights="IMAGENET1K_V1", transform_input=False)
        if num_channels == 1:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif num_channels != 3:
            raise ValueError(f'Invalid number of channels: {num_channels}')
        self.model.fc = nn.Linear(1024, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.log_softmax(x)
        return x


class ResNet(pl.LightningModule):
    def __init__(self, num_classes: int = 10, num_channels: int = 1):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        if num_channels == 1:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif num_channels != 3:
            raise ValueError(f'Invalid number of channels: {num_channels}')
        self.model.fc = nn.Linear(512, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.log_softmax(x)
        return x
