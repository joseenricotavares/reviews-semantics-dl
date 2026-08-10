from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.metrics import f1_score


def per_class_f1(y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[int]) -> dict[int, float]:
    scores = f1_score(y_true, y_pred, average=None, labels=list(classes), zero_division=0)
    return dict(zip(classes, (float(s) for s in scores), strict=True))


@dataclass(slots=True)
class SemanticDistanceReport:
    mean_distance_per_class: dict[int, float]
    error_distance_distribution: dict[int, int]


def semantic_distance_report(
    y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[int]
) -> SemanticDistanceReport:
    abs_errors = [abs(t - p) for t, p in zip(y_true, y_pred, strict=True)]

    mean_distance: dict[int, float] = {}
    for label in classes:
        errors_for_label = [e for e, t in zip(abs_errors, y_true, strict=True) if t == label]
        mean_distance[label] = sum(errors_for_label) / len(errors_for_label) if errors_for_label else 0.0

    distribution = dict(sorted(Counter(abs_errors).items()))

    return SemanticDistanceReport(
        mean_distance_per_class=mean_distance,
        error_distance_distribution=distribution,
    )
