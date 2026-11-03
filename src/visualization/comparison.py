from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_search_comparison(rows, path):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    names = [row["method"] for row in rows]
    for axis, key, title in zip(axes, ["best_val_top1", "trials", "total_seconds"], ["Best validation top-1 (%)", "Trials completed", "Total wall time (seconds)"]):
        axis.bar(names, [row[key] for row in rows])
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_model_comparison(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5))
    positions = list(range(len(rows)))
    for offset, metric in ((-0.2, "top1"), (0.2, "top5")):
        axis.bar([value + offset for value in positions], [row["metrics"][metric] for row in rows], width=0.4, label=metric)
    axis.set_xticks(positions, [row["name"] for row in rows], rotation=15)
    axis.set(ylabel="Accuracy (%)", ylim=(0, 100), title="Classification model comparison on the same split")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
