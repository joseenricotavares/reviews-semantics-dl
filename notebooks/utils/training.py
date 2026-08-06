from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal, Protocol, Sequence
from pathlib import Path
import copy

import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

class NNTrainer:
    def __init__(self, model, train_loader, val_loader, config, device, callbacks: Sequence[TrainerCallback] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.callbacks = list(callbacks or [])
        self.history: TrainingHistory = TrainingHistory()
        self.best: BestCheckpoint | None = None
        self.monitor: MonitorDefinition | None = None

    def _train_epoch(self) -> float:
        raise NotImplementedError

    def predict(self, loader):
        raise NotImplementedError

    def save(self, output_dir: str | Path) -> None:
        raise NotImplementedError
    
    def fit(self) -> TrainingHistory:
        """Trains until `config.epochs` or early stopping, returning per-epoch history."""
        for callback in self.callbacks:
            callback.on_train_begin(self)

        epochs_without_improvement = 0
        try:
            for epoch in range(1, self.config.epochs + 1):
                train_loss = self._train_epoch()
                val_loss, val_targets, val_preds = self.predict(self.val_loader)
                score = self.monitor.score(val_loss, val_targets, val_preds)

                epoch_metrics = {"train_loss": train_loss, "val_loss": val_loss}
                epoch_metrics[self.monitor.name] = score
                self.history.entries.append(epoch_metrics)

                improved = self.best is None or self.monitor.improved(score, self.best.score)
                if improved:
                    self.best = BestCheckpoint(
                        epoch=epoch, score=score,
                        state=copy.deepcopy(self.model.state_dict()),
                        targets=val_targets, preds=val_preds,
                    )
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                result = EpochResult(
                    epoch=epoch, metrics=epoch_metrics, improved=improved,
                    patience=epochs_without_improvement, best=self.best,
                )
                for callback in self.callbacks:
                    callback.on_epoch_end(self, result)

                if epochs_without_improvement >= self.config.early_stopping_patience:
                    break

            if self.best is not None:
                self.model.load_state_dict(self.best.state)
        finally:
            for callback in self.callbacks:
                callback.on_train_end(self)

        return self.history

class TrainerCallback:
    def on_train_begin(self, trainer: NNTrainer) -> None:
        """Called before the first training epoch."""

    def on_epoch_end(self, trainer: NNTrainer, result: EpochResult) -> None:
        """Called after each training epoch."""

    def on_train_end(self, trainer: NNTrainer) -> None:
        """Called after training finishes."""

@dataclass(slots=True)
class TrainingHistory:
    entries: list[dict[str, float]] = field(default_factory=list)

@dataclass(slots=True)
class EpochResult:
    epoch: int
    metrics: dict[str, float]
    improved: bool
    patience: int
    best: BestCheckpoint | None

@dataclass(slots=True)
class BestCheckpoint:
    epoch: int
    score: float
    state: dict
    targets: list
    preds: list


class MonitorMetric(str, Enum):
    VAL_LOSS = "val_loss"
    ACCURACY = "accuracy"
    F1_MACRO = "f1_macro"
    QWK = "qwk"

@dataclass(frozen=True)
class MonitorDefinition:
    name: str
    mode: Literal["min", "max"]
    scorer: Callable[[list, list], float] | None = None

    def score(self, val_loss: float, y_true: list, y_pred: list) -> float:
        if self.scorer is None:
            return val_loss
        return self.scorer(y_true, y_pred)

    def improved(self, current: float, best: float) -> bool:
        return current > best if self.mode == "max" else current < best


MONITORS: dict[MonitorMetric, MonitorDefinition] = {
    MonitorMetric.VAL_LOSS: MonitorDefinition(
        name="val_loss",
        mode="min",
    ),
    MonitorMetric.ACCURACY: MonitorDefinition(
        name="val_accuracy",
        mode="max",
        scorer=accuracy_score,
    ),
    MonitorMetric.F1_MACRO: MonitorDefinition(
        name="val_f1_macro",
        mode="max",
        scorer=lambda y_true, y_pred: f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    ),
    MonitorMetric.QWK: MonitorDefinition(
        name="val_qwk",
        mode="max",
        scorer=lambda y_true, y_pred: cohen_kappa_score(
            y_true,
            y_pred,
        ),
    ),
}

class OptimizerType(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"

OPTIMIZERS: dict[
    OptimizerType,
    Callable[..., optim.Optimizer],
] = {
    OptimizerType.ADAM: optim.Adam,
    OptimizerType.ADAMW: optim.AdamW,
    OptimizerType.SGD: lambda params, lr, weight_decay: optim.SGD(
        params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9,
    ),
}

