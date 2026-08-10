from __future__ import annotations

from dataclasses import dataclass

from dlkit.optim import OptimizerType
from dlkit.training import Monitor


@dataclass
class SearchSpaceConfig:
    """Optuna hyperparameter-search space for the BiLSTM baseline."""

    hidden_dim_options: tuple[int, ...]
    dropout_range: tuple[float, float]
    lr_range: tuple[float, float]
    weight_decay_range: tuple[float, float]

    epochs: int
    optimizer: OptimizerType
    monitor: Monitor
    early_stopping_patience: int
