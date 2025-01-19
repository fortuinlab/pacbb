"""
The `split_strategy` subpackage defines how datasets are split among
prior training, posterior training, bound evaluation, and validation/testing.

It includes:
  - `AbstractSplitStrategy` as an interface,
  - `PBPSplitStrategy`, `FaultySplitStrategy`, or other implementations
    that partition data for different PAC-Bayes workflows.

Use these strategies to ensure clean separation of data and avoid leakage
between prior, posterior, and bound sets.
"""

from core.split_strategy.AbstractSplitStrategy import AbstractSplitStrategy
from core.split_strategy.PBPSplitStrategy import PBPSplitStrategy
from core.split_strategy.FaultySplitStrategy import FaultySplitStrategy
