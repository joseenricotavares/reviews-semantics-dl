from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error


@dataclass(slots=True)
class MAEMetric:
    name: str = "mae"

    def compute(self, y_true: Sequence, y_pred: Sequence) -> float:
        return float(mean_absolute_error(y_true, y_pred))


@dataclass(slots=True)
class RMSEMetric:
    name: str = "rmse"

    def compute(self, y_true: Sequence, y_pred: Sequence) -> float:
        return float(mean_squared_error(y_true, y_pred) ** 0.5)


@dataclass(slots=True)
class QuadraticWeightedKappaMetric:
    name: str = "qwk"

    def compute(self, y_true: Sequence, y_pred: Sequence) -> float:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
