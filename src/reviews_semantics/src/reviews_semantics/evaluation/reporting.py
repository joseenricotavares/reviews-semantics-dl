from __future__ import annotations


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
