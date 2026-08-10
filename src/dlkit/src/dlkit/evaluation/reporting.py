from __future__ import annotations

from collections.abc import Sequence


def build_confusion_matrix_figure(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
    *,
    title: str = "Confusion Matrix",
):
    """Returns a matplotlib `Figure` with a confusion-matrix heatmap."""
    import matplotlib

    matplotlib.use("Agg")  
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    labels = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            ax.text(
                j, i, str(value), ha="center", va="center",
                color="white" if value > threshold else "black",
            )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    return fig


def build_training_curve_figure(
    entries: Sequence[dict[str, float]],
    metric: str,
    *,
    title: str = "Training History",
):
    """Returns a matplotlib `Figure` plotting train/val loss and `metric` per epoch.

    `entries` is the raw list of per-epoch metric dicts (e.g.
    `TrainingHistory.entries`), each expected to contain `train_loss`,
    `val_loss`, and `metric`.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: this module never renders interactively
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = list(range(1, len(entries) + 1))
    train_loss = [e["train_loss"] for e in entries]
    val_loss = [e["val_loss"] for e in entries]
    metric_values = [e[metric] for e in entries]

    width = min(max(8, len(entries) * 0.25), 20)
    fig, ax_loss = plt.subplots(figsize=(width, 4))
    ax_metric = ax_loss.twinx()

    line_kwargs = {"linewidth": 2, "marker": "o", "markersize": 3}
    ax_loss.plot(epochs, train_loss, label="Train Loss", color="tab:blue", **line_kwargs)
    ax_loss.plot(epochs, val_loss, label="Validation Loss", color="tab:cyan", **line_kwargs)
    ax_metric.plot(epochs, metric_values, label=metric, color="tab:orange", **line_kwargs)

    ax_loss.set(title=title, xlabel="Epoch", ylabel="Loss")
    ax_metric.set_ylabel(metric)
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_loss.grid(True)

    lines = ax_loss.get_lines() + ax_metric.get_lines()
    ax_loss.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    return fig
