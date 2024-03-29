from torch import nn

from core.layer import ProbLinear, ProbConv2d

LAYER_MAPPING = {
    nn.Linear: ProbLinear,
    nn.Conv2d: ProbConv2d,
}
