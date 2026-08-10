from __future__ import annotations

from typing import Protocol, runtime_checkable

from dlkit.training.monitor import Monitor


@runtime_checkable
class TrainerConfig(Protocol):
    """The minimal shape `NNTrainer.fit()` needs from a config object."""

    epochs: int
    early_stopping_patience: int
    monitor: Monitor
