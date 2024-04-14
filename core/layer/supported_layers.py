from torch import nn

from core.layer import ProbLinear, ProbConv2d, ProbBatchNorm1d, ProbBatchNorm2d

LAYER_MAPPING = {
    nn.Linear: ProbLinear,
    nn.Conv2d: ProbConv2d,
    nn.BatchNorm1d: ProbBatchNorm1d,
    nn.BatchNorm2d: ProbBatchNorm2d,
}
