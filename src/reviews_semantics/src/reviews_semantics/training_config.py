from __future__ import annotations

from dataclasses import dataclass, field

from dlkit.evaluation import AccuracyMetric, F1Metric
from dlkit.optim import OptimizerType
from dlkit.training import Monitor

from reviews_semantics.evaluation.ordinal_metrics import QuadraticWeightedKappaMetric


def _default_monitor() -> Monitor:
    return Monitor(metric=F1Metric("macro"), mode="max")


@dataclass
class TrainingConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    optimizer: OptimizerType = OptimizerType.ADAMW
    grad_clip_norm: float | None = 5.0
    monitor: Monitor = field(default_factory=_default_monitor)
    early_stopping_patience: int = 5


STAR_MONITORS: dict[str, Monitor] = {
    "val_loss": Monitor(metric=None, mode="min"),
    "accuracy": Monitor(metric=AccuracyMetric(), mode="max"),
    "f1_macro": Monitor(metric=F1Metric("macro"), mode="max"),
    "qwk": Monitor(metric=QuadraticWeightedKappaMetric(), mode="max"),
}
