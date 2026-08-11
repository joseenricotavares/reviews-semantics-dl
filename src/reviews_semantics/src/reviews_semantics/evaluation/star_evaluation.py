from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dlkit.evaluation import AccuracyMetric, EvaluationResult, Evaluator, F1Metric

from reviews_semantics.evaluation.ordinal_metrics import MAEMetric, QuadraticWeightedKappaMetric, RMSEMetric
from reviews_semantics.evaluation.reports import (
    SemanticDistanceReport,
    per_class_f1,
    semantic_distance_report,
)
from reviews_semantics.labels import STAR_SCHEMA

_EVALUATOR = Evaluator(
    [
        AccuracyMetric(),
        F1Metric("macro"),
        MAEMetric(),
        RMSEMetric(),
        QuadraticWeightedKappaMetric(),
    ]
)


@dataclass(slots=True)
class StarEvaluation:
    scalars: EvaluationResult
    per_class_f1: dict[int, float]
    semantic_distance: SemanticDistanceReport


def evaluate_stars(y_true: Sequence[int], y_pred: Sequence[int]) -> StarEvaluation:
    return StarEvaluation(
        scalars=_EVALUATOR.evaluate(y_true, y_pred),
        per_class_f1=per_class_f1(y_true, y_pred, STAR_SCHEMA.classes),
        semantic_distance=semantic_distance_report(y_true, y_pred, STAR_SCHEMA.classes),
    )
