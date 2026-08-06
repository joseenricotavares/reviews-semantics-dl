import sys
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

@dataclass(slots=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    class_f1: Dict[int, float]

@dataclass(slots=True)
class OrdinalMetrics:
    mae: float
    rmse: float
    qwk: float

@dataclass(slots=True)
class SemanticDistanceMetrics:
    mean_distance_per_class: Dict[int, float]
    error_distance_distribution: Dict[int, int]

@dataclass(slots=True)
class EvaluationResult:
    classification: ClassificationMetrics
    ordinal: OrdinalMetrics
    semantic: SemanticDistanceMetrics

class EvaluationPhase(Enum):
    VALIDATION = "Validation Data"
    TEST = "Test Data"

    @property
    def prefix(self) -> str:
        return "val" if self is EvaluationPhase.VALIDATION else "test"


class MetricsEvaluator:

    def __init__(self, class_labels: Sequence[int] = (1, 2, 3, 4, 5)) -> None:
        self.class_labels = list(class_labels)

    def evaluate(self, model_name: str, y_true: Sequence[int], y_pred: Sequence[int], phase: EvaluationPhase) -> EvaluationResult:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        classification = self._classification_metrics(y_true, y_pred)
        ordinal = self._ordinal_metrics(y_true, y_pred)
        semantic = self._semantic_distance_metrics(y_true, y_pred)
        result = EvaluationResult(
            classification=classification,
            ordinal=ordinal,
            semantic=semantic,
        )
        self._log_metrics(model_name, result, phase)
        self._save_artifacts(model_name, y_true, y_pred, result, phase)
        return result

    def _classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
        class_scores = f1_score(y_true, y_pred, average=None, labels=self.class_labels, zero_division=0)
        return ClassificationMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
            class_f1=dict(zip(self.class_labels, class_scores)),
        )

    def _ordinal_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> OrdinalMetrics:
        return OrdinalMetrics(
            mae=mean_absolute_error(y_true, y_pred),
            rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
            qwk=cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    def _semantic_distance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> SemanticDistanceMetrics: 
        abs_errors = np.abs(y_true - y_pred)
        mean_distance = {}
        for label in self.class_labels:
            mask = y_true == label
            mean_distance[label] = (float(abs_errors[mask].mean()) if mask.any() else 0.0)
        distribution = (pd.Series(abs_errors).value_counts().sort_index().to_dict())
        return SemanticDistanceMetrics(
            mean_distance_per_class=mean_distance,
            error_distance_distribution=distribution,
        )

    def _log_metrics(self, model_name: str, result: EvaluationResult, phase: EvaluationPhase) -> None:
        p = phase.prefix
        # Classification Metrics
        mlflow.log_metric(f"{p}_accuracy", result.classification.accuracy)
        mlflow.log_metric(f"{p}_macro_f1", result.classification.macro_f1)
        for label, score in result.classification.class_f1.items():
            mlflow.log_metric(f"{p}_f1_class_{label}_stars", score)

        # Ordinal Metrics
        mlflow.log_metric(f"{p}_mae", result.ordinal.mae)
        mlflow.log_metric(f"{p}_rmse", result.ordinal.rmse)
        mlflow.log_metric(f"{p}_qwk", result.ordinal.qwk)

        # Semantic Distance Metrics
        for label, distance in result.semantic.mean_distance_per_class.items():
            mlflow.log_metric(f"{p}_mean_error_distance_class_{label}", distance)
        for distance, count in result.semantic.error_distance_distribution.items():
            mlflow.log_metric(f"{p}_error_distance_{distance}", count)

        #self._print_summary(model_name, result)

    def _save_artifacts(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, result: EvaluationResult, phase: EvaluationPhase) -> None:
        stem = f"{model_name.replace(' ', '_')}_{phase.name.lower()}"
        predictions_path = Path(f"predictions_{stem}.csv")
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(predictions_path, index=False)
        mlflow.log_artifact(predictions_path)

        path = Path(f"evaluation_{stem}.png")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [1.35, 1.0, 0.75]})
        fig.suptitle(f"Evaluation on {phase.value}: {model_name}", fontsize=16, fontweight="bold")
        self._plot_confusion_matrix(model_name, y_true, y_pred, axes[0])
        self._plot_error_distance(model_name, result.semantic.error_distance_distribution, axes[1])
        self._plot_summary(result, axes[2])
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        mlflow.log_artifact(path)
        plt.show() 
        plt.close(fig)
        path.unlink()
        predictions_path.unlink()

    def _plot_confusion_matrix(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, ax: plt.Axes) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=self.class_labels)
        sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=self.class_labels, yticklabels=self.class_labels, ax=ax)
        threshold = cm.max() / 2
        for i, row in enumerate(cm):
            for j, value in enumerate(row):
                ax.text(j + 0.5, i + 0.5, value, ha="center", va="center", color="white" if value > threshold else "black")
        ax.set(title=f"Confusion Matrix - {model_name}", xlabel="Predicted", ylabel="Actual")

    def _plot_error_distance(self, model_name: str, distribution: Dict[int, int], ax: plt.Axes) -> None:
        df = pd.DataFrame(distribution.items(), columns=["Distance", "Count"]).sort_values("Distance")
        total = df["Count"].sum()
        max_count = df["Count"].max()

        ax.barh(df["Distance"], df["Count"], color="#c2185b", edgecolor="black", height=0.8)
        for idx, count in enumerate(df["Count"]):
            ax.text(count + max_count * 0.02, idx, f"{count}\n({count / total:.1%})",
                    ha="left", va="center", fontsize=10, fontweight="bold")

        ax.set(title=f"Error Distance Distribution - {model_name}",
            xlabel="Number of Predictions",
            ylabel="Absolute Error Distance (in Stars)")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["Distance"].astype(int).tolist())
        ax.set_xlim(0, max_count * 1.15)
        ax.grid(axis="x", alpha=0.3)
        sns.despine()

    def _plot_summary(self, result: EvaluationResult, ax: plt.Axes) -> None:
        ax.axis("off")
        y = 0.98 
        
        def section(title):
            nonlocal y
            if y < 0.98:
                ax.plot([0.0, 0.95], [y + 0.02, y + 0.02], color="gray", lw=0.5, alpha=0.3, transform=ax.transAxes)
                y -= 0.015
            ax.text(0.0, y, title, fontsize=11, fontweight="bold", color="#2c3e50", transform=ax.transAxes)
            y -= 0.055

        def metric(name, value):
            nonlocal y
            ax.text(0.05, y, name, fontsize=10, color="#34495e", transform=ax.transAxes)
            ax.text(0.95, y, f"{value:.4f}", fontsize=10, family="monospace", ha="right", transform=ax.transAxes)
            y -= 0.045 

        section("Target Metric")
        metric("Macro F1", result.classification.macro_f1)
        y -= 0.01 
        section("Classification")
        metric("Accuracy", result.classification.accuracy)
        for label, score in result.classification.class_f1.items():
            metric(f"F1 ({label}★)", score)
        y -= 0.01
        section("Ordinal")
        metric("MAE", result.ordinal.mae)
        metric("RMSE", result.ordinal.rmse)
        metric("QWK", result.ordinal.qwk)
        y -= 0.01
        section("Mean Error Distance")
        for label, distance in result.semantic.mean_distance_per_class.items():
            metric(f"{label}★", distance)

    def _print_summary(self, model_name: str, result: EvaluationResult) -> None:
        print("=" * 60)
        print(model_name)
        print("=" * 60)
        print("\nTarget Metric")
        print(f"Macro F1 : {result.classification.macro_f1:.4f}")
        print("\nClassification")
        print(f"Accuracy : {result.classification.accuracy:.4f}")
        for label, score in result.classification.class_f1.items():
            print(f"F1 ({label}★): {score:.4f}")
        print("\nOrdinal")
        print(f"MAE  : {result.ordinal.mae:.4f}")
        print(f"RMSE : {result.ordinal.rmse:.4f}")
        print(f"QWK  : {result.ordinal.qwk:.4f}")
        print("\nSemantic Distance")
        for label, distance in result.semantic.mean_distance_per_class.items():
            print(f"Mean Error Distance ({label}★): {distance:.4f}")