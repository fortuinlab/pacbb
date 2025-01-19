"""
The `core` package provides foundational components for the PAC-Bayes framework.

It includes:
  - Subpackages for distribution modeling (`distribution/`),
    probabilistic layers (`layer/`), PAC-Bayes bounds (`bound/`),
    objectives (`objective/`), data splitting strategies (`split_strategy/`),
    and general utilities (`utils/`).
  - Higher-level modules (`loss.py`, `metric.py`, `model.py`, `risk.py`,
    `training.py`) that stitch these components together.

By assembling the pieces within `core`, users can build, train, and evaluate
probabilistic neural networks under the PAC-Bayes paradigm.
"""
