from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from dlkit.training.callbacks import build as build_callbacks
from torch.utils.data import DataLoader

from reviews_semantics.data.dataset import ReviewDataset
from reviews_semantics.data.splits import load_split
from reviews_semantics.data.vocab import TextPreprocessor
from reviews_semantics.paradigms.bilstm.model import SentimentLSTM
from reviews_semantics.paradigms.bilstm.trainer import ClassifierTrainer
from reviews_semantics.training_config import TrainingConfig

NUM_CLASSES = 5


def train_baseline(
    *,
    data_dir: Path,
    output_dir: Path,
    config: TrainingConfig | None = None,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    max_len: int = 50,
    max_vocab_size: int = 10000,
    batch_size: int = 64,
    device: torch.device | None = None,
    enable_mlflow: bool = True,
    mlflow_experiment_name: str = "Olist_Review_Classification",
    mlflow_tracking_uri: str | None = None,
    run_name: str = "Baseline_BiLSTM",
) -> ClassifierTrainer:
    """Trains the BiLSTM baseline end to end and writes its artifact bundle to `output_dir`.

    A plain, non-notebook equivalent of the source notebook's baseline
    training section (data loading through `train_and_log()`), so the exact
    same paradigm/hyperparameters can be reproduced from the training
    container's CLI.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config or TrainingConfig()

    X_train, y_train_stars = load_split(Path(data_dir) / "train.csv")
    X_val, y_val_stars = load_split(Path(data_dir) / "val.csv")
    y_train = [score - 1 for score in y_train_stars]  # 0-index for nn.CrossEntropyLoss
    y_val = [score - 1 for score in y_val_stars]

    preprocessor = TextPreprocessor.build(X_train, max_len=max_len, max_vocab_size=max_vocab_size)

    train_loader = DataLoader(
        ReviewDataset(X_train, y_train, preprocessor), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        ReviewDataset(X_val, y_val, preprocessor), batch_size=batch_size, shuffle=False
    )

    model = SentimentLSTM(
        vocab_size=len(preprocessor.vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=NUM_CLASSES,
        padding_idx=preprocessor.vocab["<PAD>"],
    )

    run_context = nullcontext()
    if enable_mlflow:
        import mlflow

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        tags = {"architecture": "BiLSTM", "stage": "baseline"}
        run_context = mlflow.start_run(run_name=run_name, tags=tags)

    with run_context:
        if enable_mlflow:
            import mlflow

            mlflow.log_params(
                {
                    "epochs": config.epochs,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "optimizer": config.optimizer.value,
                    "grad_clip_norm": config.grad_clip_norm,
                    "monitor": config.monitor.name,
                    "early_stopping_patience": config.early_stopping_patience,
                }
            )

        trainer = ClassifierTrainer(
            model, train_loader, val_loader, config, preprocessor, device,
            callbacks=build_callbacks(enable_mlflow=enable_mlflow),
        )
        trainer.fit()

    trainer.save(output_dir)
    return trainer
