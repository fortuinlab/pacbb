"""
## Overview
Defines how to split datasets into parts for prior training, posterior
training, bound evaluation, validation, and testing in PAC-Bayes pipelines.

## Contents
- **AbstractSplitStrategy**: The base interface
- **PBPSplitStrategy, FaultySplitStrategy**: Concrete implementations
  to partition data for different training/evaluation scenarios

Use these strategies to ensure data for prior and posterior does not overlap
and to reserve a portion for bound computation.
"""

from core.split_strategy.AbstractSplitStrategy import AbstractSplitStrategy
from core.split_strategy.PBPSplitStrategy import PBPSplitStrategy
from core.split_strategy.FaultySplitStrategy import FaultySplitStrategy
