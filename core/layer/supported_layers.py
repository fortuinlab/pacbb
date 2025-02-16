from torch import nn

from core.layer import ProbBatchNorm1d, ProbBatchNorm2d, ProbConv2d, ProbLinear

LAYER_MAPPING = {
    nn.Linear: ProbLinear,
    nn.Conv2d: ProbConv2d,
    nn.BatchNorm1d: ProbBatchNorm1d,
    nn.BatchNorm2d: ProbBatchNorm2d,
}
