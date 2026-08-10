# dlkit

Reusable building blocks extracted from doing epoch-based PyTorch training with
MLflow/Optuna/rich tooling more than once: a training loop plus a callback
system, and a small, framework-agnostic contract for packaging and serving
whatever a training run produces.

`dlkit` is deliberately **not** a universal ML framework. It exists to avoid
re-writing the same training loop and the same "how do I get this model into
an API" plumbing on every new project.

## What it is built for

- **`dlkit.training` / `dlkit.optim` / `dlkit.callbacks`** — PyTorch. An
  epoch-based `NNTrainer` (train/validate/early-stop/checkpoint loop) that any
  `torch.nn.Module`-based model can plug into by implementing three methods.
- **`dlkit.evaluation`** — small, composable evaluation primitives
  (`Metric` + `Evaluator`) built on scikit-learn, plus a couple of metrics
  that are generic to any classification problem (accuracy, F1).
- **`dlkit.artifacts` / `dlkit.inference`** — framework-agnostic. A model
  artifact bundle format and a `Predictor` protocol that work equally well
  for a PyTorch model or a plain scikit-learn pipeline, as long as whatever
  produced it can write the bundle and whatever serves it can implement
  `predict()`.

## Explicit non-goals

These are deliberate scope boundaries, not oversights:

- No attempt to generalize non-epoch training loops (e.g. scikit-learn's
  one-shot `.fit()`) into `NNTrainer`. If a model doesn't train in epochs, it
  doesn't need `dlkit.training` at all — it only needs to satisfy the
  `dlkit.artifacts`/`dlkit.inference` contracts to be servable.
- No built-in regression or multi-label support. `dlkit.artifacts.LabelSchema`
  assumes discrete classes.
- No attempt to cover streaming or online learning.
- No fixed evaluation taxonomy. `dlkit.evaluation` gives you `Metric` + `Evaluator`
  and a couple of ready-made generic metrics; anything domain-specific is expected to
  be implemented by the consuming project as its own `Metric`.

## What already works without any dlkit change

Binary and multiclass classification both work out of the box: nothing in
`dlkit.training` or `dlkit.evaluation` assumes a fixed number of classes or a
fixed set of metrics — the `Monitor`/`Metric` abstractions accept any
`Metric` implementation, including ones defined entirely outside `dlkit`.

## Installing

```bash
pip install -e src/dlkit                                   # base: NNTrainer, artifacts, inference, generic evaluation
pip install -e "src/dlkit[callbacks-mlflow,plotting]"       # add extras as needed
```

Extras: `callbacks-mlflow`, `callbacks-rich`, `callbacks-optuna`, `plotting`.
Each isolates its heavy/optional dependency (`mlflow`, `rich`, `optuna`,
`matplotlib`) so a lean serving environment never has to install them.

## Minimal example

```python
from dlkit.training import NNTrainer, TrainerConfig, Monitor
from dlkit.evaluation import AccuracyMetric

class MyTrainer(NNTrainer):
    def _train_epoch(self) -> float: ...
    def predict(self, loader): ...
    def save(self, output_dir): ...

config = TrainerConfig(
    epochs=10,
    early_stopping_patience=3,
    monitor=Monitor(metric=AccuracyMetric(), mode="max"),
    optimizer="adamw",
    learning_rate=1e-3,
    weight_decay=1e-2,
)
```
