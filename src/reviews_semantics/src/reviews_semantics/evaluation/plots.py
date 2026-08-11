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


def build_semantic_distance_figure(
    distribution: dict[int, int], *, title: str = "Error Distance Distribution"
):
    """Returns a matplotlib `Figure`: a horizontal bar chart of how many
    predictions landed at each absolute star-distance from the true label.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distances = sorted(distribution)
    counts = [distribution[d] for d in distances]
    total = sum(counts)
    max_count = max(counts) if counts else 0

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(range(len(distances)), counts, color="#c2185b", edgecolor="black", height=0.8)
    for idx, count in enumerate(counts):
        pct = count / total if total else 0.0
        ax.text(
            count + max_count * 0.02, idx, f"{count}\n({pct:.1%})",
            ha="left", va="center", fontsize=10, fontweight="bold",
        )

    ax.set(title=title, xlabel="Number of Predictions", ylabel="Absolute Error Distance (in Stars)")
    ax.set_yticks(range(len(distances)))
    ax.set_yticklabels([str(d) for d in distances])
    ax.set_xlim(0, max_count * 1.15 if max_count else 1)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
