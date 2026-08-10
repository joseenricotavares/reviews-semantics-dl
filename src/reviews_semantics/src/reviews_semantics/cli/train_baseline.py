from __future__ import annotations

from pathlib import Path

from reviews_semantics.paradigms.bilstm.train import train_baseline
from reviews_semantics.paradigms.bilstm.trainer import ClassifierTrainer
from reviews_semantics.training_config import TrainingConfig


def run_train_baseline(
    *,
    data_dir: Path,
    output_dir: Path,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    enable_mlflow: bool = True,
) -> ClassifierTrainer:
    """CLI-facing wrapper over `paradigms.bilstm.train.train_baseline`, used
    as the training container's entrypoint."""
    config = TrainingConfig(epochs=epochs, learning_rate=learning_rate)
    return train_baseline(
        data_dir=data_dir, output_dir=output_dir, config=config, enable_mlflow=enable_mlflow
    )
