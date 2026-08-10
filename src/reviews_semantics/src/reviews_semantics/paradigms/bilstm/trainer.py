from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
from dlkit.artifacts import ArtifactBundle, BundleFile
from dlkit.training import OPTIMIZERS, NNTrainer, TrainerCallback
from torch.utils.data import DataLoader

from reviews_semantics.data.vocab import TextPreprocessor
from reviews_semantics.labels import STAR_SCHEMA
from reviews_semantics.training_config import TrainingConfig


class ClassifierTrainer(NNTrainer[TrainingConfig]):

    def __init__(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        config: TrainingConfig, preprocessor: TextPreprocessor, device: torch.device,
        callbacks: Sequence[TrainerCallback] | None = None,
    ) -> None:
        super().__init__(model, train_loader, val_loader, config, device, callbacks)
        self.criterion = nn.CrossEntropyLoss()  # hardcoded to guarantee the model outputs raw logits
        self.optimizer = OPTIMIZERS[config.optimizer](
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.preprocessor = preprocessor

    def _train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()

            if self.config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[float, list, list]:
        """Runs inference over `loader`, returning (avg_loss, targets, predictions)."""
        self.model.eval()
        running_loss = 0.0
        targets: list[int] = []
        preds: list[int] = []
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            logits = self.model(X_batch)
            running_loss += self.criterion(logits, y_batch).item()
            preds.extend((torch.argmax(logits, dim=1) + 1).cpu().tolist())
            targets.extend((y_batch + 1).cpu().tolist())
        return running_loss / len(loader), targets, preds

    def _write_model_config(self, path: Path) -> None:
        path.write_text(json.dumps(self.model.export_config(), indent=4), encoding="utf-8")

    def _write_preprocessor(self, path: Path) -> None:
        data = json.dumps(self.preprocessor.to_dict(), ensure_ascii=False, indent=4)
        path.write_text(data, encoding="utf-8")

    def save(self, output_dir: str | Path) -> ArtifactBundle:
        assert self.best is not None, "save() called before fit() produced a best checkpoint"
        best_state = self.best.state
        return ArtifactBundle.write(
            output_dir,
            flavor="bilstm",
            label_schema=STAR_SCHEMA,
            files={
                "weights": BundleFile("state_dict.pt", lambda p: torch.save(best_state, p)),
                "model_config": BundleFile("model.json", self._write_model_config),
                "preprocessor": BundleFile("preprocessor.json", self._write_preprocessor),
            },
            metrics={self.monitor.name: self.best.score},
        )
