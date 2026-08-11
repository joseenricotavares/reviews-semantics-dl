from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
from dlkit.artifacts import ArtifactBundle, BundleFile
from dlkit.training import OPTIMIZERS, NNTrainer, TrainerCallback
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from reviews_semantics.labels import STAR_SCHEMA
from reviews_semantics.training_config import TrainingConfig


class TransformerTrainer(NNTrainer[TrainingConfig]):
    """Fine-tunes a HuggingFace sequence-classification model."""

    def __init__(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        config: TrainingConfig, tokenizer, device: torch.device,
        callbacks: Sequence[TrainerCallback] | None = None, num_warmup_steps: int | None = None,
    ) -> None:
        super().__init__(model, train_loader, val_loader, config, device, callbacks)
        self.tokenizer = tokenizer
        self.optimizer = OPTIMIZERS[config.optimizer](
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        # Linear warmup + decay over the full (max) training horizon.
        total_steps = len(train_loader) * config.epochs
        warmup = num_warmup_steps if num_warmup_steps is not None else int(0.1 * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
        )

    def _train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # cross-entropy computed internally
            loss.backward()

            if self.config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

            self.optimizer.step()
            self.scheduler.step()  # per-batch step for linear warmup/decay
            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[float, list, list]:
        """Runs inference over `loader`, returning (avg_loss, targets, predictions) in the 1-5 star range."""
        self.model.eval()
        running_loss = 0.0
        targets: list[int] = []
        preds: list[int] = []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            running_loss += outputs.loss.item()
            preds.extend((torch.argmax(outputs.logits, dim=1) + 1).cpu().tolist())
            targets.extend((labels + 1).cpu().tolist())

        return running_loss / len(loader), targets, preds

    def save(self, output_dir: str | Path) -> ArtifactBundle:
        output_dir = Path(output_dir)

        def _save_checkpoint(path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

        return ArtifactBundle.write(
            output_dir,
            flavor="transformer_finetune",
            label_schema=STAR_SCHEMA,
            files={"hf_checkpoint": BundleFile("hf_checkpoint", _save_checkpoint)},
            metrics={self.monitor.name: self.best.score} if self.best is not None else {},
        )
