import shutil
from pathlib import Path
import random
import time
import io
from IPython.display import display, Image

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mlflow
import pandas as pd

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

import optuna

from .training import NNTrainer, EpochResult, TrainerCallback


class MLflowCallback(TrainerCallback):
    """Logs training metrics to the active MLflow run."""
    def __init__(self, artifact_dir: str = "model") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.start_time: float | None = None

    def on_train_begin(self, trainer: NNTrainer) -> None:
        self.start_time = time.perf_counter()

    def on_epoch_end(self, trainer: NNTrainer, result: EpochResult) -> None:
        mlflow.log_metrics(result.metrics, step=result.epoch)
    
    def on_train_end(self, trainer: NNTrainer) -> None:
        if self.start_time is not None:
            train_time = time.perf_counter() - self.start_time
            mlflow.log_metric("train_time_seconds", train_time)
        if trainer.best is None:
            return
        trainer.save(self.artifact_dir)
        try:
            mlflow.log_artifacts(local_dir=str(self.artifact_dir),artifact_path="model")
        finally:
            shutil.rmtree(self.artifact_dir, ignore_errors=True)

class RichProgressCallback(TrainerCallback):
    """Displays training progress using a live Rich table."""

    def __init__(self) -> None:
        self.console = Console()
        self.live: Live | None = None
        self.last_table = None  

    def on_train_begin(self, trainer: NNTrainer) -> None:
        self.live = Live(console=self.console, refresh_per_second=4)
        self.live.start()

    def on_epoch_end(self, trainer: NNTrainer, result: EpochResult) -> None:
        assert self.live is not None
        self.last_table = self._build_table(trainer, result)
        self.live.update(self.last_table)

    def on_train_end(self, trainer: NNTrainer) -> None:
        if self.live is not None:
            self.live.stop()
        if self.last_table is not None:
            self.console.print(self.last_table)
            
    @staticmethod
    def _format_value(value: object) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    def _build_table(self, trainer: NNTrainer, result: EpochResult) -> Panel:
        table = Table.grid(padding=(0, 2))
        metric_name = trainer.monitor.name
        best_score = self._format_value(result.best.score) if result.best is not None else "n/a"
        if result.improved:
            best_score += " ✓"

        table.add_column(style="cyan")
        table.add_column(justify="right")
        table.add_row("Epoch",f"{result.epoch}/{trainer.config.epochs}")
        table.add_row("Train Loss", self._format_value(result.metrics['train_loss']))
        table.add_row("Val Loss", self._format_value(result.metrics['val_loss']))
        table.add_row(metric_name, self._format_value(result.metrics[metric_name]))
        table.add_row(f"Best {metric_name}", best_score)
        table.add_row("Best Epoch", str(result.best.epoch) if result.best is not None else "n/a")
        table.add_row("Patience",f"{result.patience}/{trainer.config.early_stopping_patience}")
        return Panel(table, title="Training Progress")

class HistoryPlotCallback(TrainerCallback):
    """Plots the training history and logs it as an MLflow artifact."""

    def __init__(self, keep_local: bool = False) -> None:
        suffix = random.randint(1000, 9999)
        self.filename = Path(f"training_history_{suffix}.png")
        self.keep_local = keep_local

    def on_train_end(self, trainer: NNTrainer) -> None:
        if not trainer.history.entries:
            return
        history = pd.DataFrame(trainer.history.entries)
        history.insert(0, "epoch", range(1, len(history) + 1))
        fig = self._make_figure(history, trainer.monitor.name)
        self._display_inline(fig)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        csv_path = self.filename.with_suffix(".csv")
        try:
            fig.savefig(self.filename, dpi=300, bbox_inches="tight")
            history.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(self.filename))
            mlflow.log_artifact(str(csv_path))
        finally:
            plt.close(fig)

            if not self.keep_local:
                for path in (self.filename, csv_path):
                    if path.exists():
                        path.unlink()

    @staticmethod
    def _display_inline(fig: plt.Figure) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        width = int(fig.get_figwidth() * 100) 
        display(Image(data=buf.getvalue(), width=width))

    def _make_figure(self, history: pd.DataFrame, metric: str) -> plt.Figure:
        n_epochs = len(history)
        width = min(max(8, n_epochs * 0.25), 20)
        fig, ax_loss = plt.subplots(figsize=(width, 4))
        ax_metric = ax_loss.twinx()
        ax_loss.plot(history["epoch"], history["train_loss"], label="Train Loss", color="tab:blue", linewidth=2, marker="o", markersize=3)
        ax_loss.plot(history["epoch"], history["val_loss"], label="Validation Loss", color="tab:cyan", linewidth=2, marker="o", markersize=3)
        ax_metric.plot(history["epoch"], history[metric], label=metric, linewidth=2, color="tab:orange", marker="o", markersize=3)
        ax_loss.set(title="Training History", xlabel="Epoch", ylabel="Loss")
        ax_metric.set_ylabel(metric)
        ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_loss.grid(True)
        lines = ax_loss.get_lines() + ax_metric.get_lines()
        ax_loss.legend(lines, [line.get_label() for line in lines], loc="best")
        fig.tight_layout()
        return fig

class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial: optuna.Trial) -> None:
        self.trial = trial

    def on_epoch_end(self, trainer, result) -> None:
        self.trial.report(result.metrics[trainer.monitor.name], step=result.epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def build_callbacks(
    *, mlflow: bool = True, progress: bool = True, plot: bool = True, optuna: bool = False,
    keep_plot: bool = False,
) -> list[TrainerCallback]:
    callbacks: list[TrainerCallback] = []

    if mlflow:
        callbacks.append(MLflowCallback())
    if progress:
        callbacks.append(RichProgressCallback())
    if plot:
        callbacks.append(HistoryPlotCallback(keep_plot))
    if optuna:
        callbacks.append(OptunaPruningCallback())

    return callbacks