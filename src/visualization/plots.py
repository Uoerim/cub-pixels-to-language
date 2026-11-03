import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def denormalize(image):
    values = image.detach().cpu()
    mean = torch.tensor(IMAGENET_MEAN, dtype=values.dtype)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=values.dtype)[:, None, None]
    return (values * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def save_training_curves(history, output_dir="outputs/figures", prefix="scratch"):
    if not history:
        raise ValueError("Training history is empty.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    loss_path = output_dir / f"loss_{prefix}.png"
    figure, axis = plt.subplots(figsize=(8, 5))
    for split, label in (("train", "Training"), ("val", "Validation")):
        axis.plot(epochs, [row[f"{split}_loss"] for row in history], label=label)
    axis.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Classification loss")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(loss_path, dpi=160)
    plt.close(figure)
    accuracy_path = output_dir / f"accuracy_{prefix}.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric in zip(axes, ("top1", "top5")):
        for split, label in (("train", "Training"), ("val", "Validation")):
            axis.plot(epochs, [row[f"{split}_{metric}"] for row in history], label=label)
        axis.set(xlabel="Epoch", ylabel="Accuracy (%)", title=f"Top-{metric[-1]} accuracy", ylim=(0, 100))
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(accuracy_path, dpi=160)
    plt.close(figure)
    return [loss_path, accuracy_path]


def save_augmented_samples(dataset, output_dir="outputs/figures", num_samples=8, prefix="scratch"):
    count = min(num_samples, len(dataset))
    if count < 1:
        raise ValueError("At least one image is needed for the augmentation plot.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = min(4, count)
    figure, axes = plt.subplots(math.ceil(count / columns), columns, figsize=(columns * 3, math.ceil(count / columns) * 3), squeeze=False)
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    names = getattr(dataset, "class_names", [])
    for axis, index in zip(axes.flat, indices):
        image, label = dataset[int(index)]
        axis.imshow(denormalize(image))
        axis.set_title(names[label] if label < len(names) else f"Class {label}", fontsize=9)
        axis.axis("off")
    for axis in list(axes.flat)[count:]:
        axis.axis("off")
    figure.suptitle("Training augmentations")
    figure.tight_layout()
    path = output_dir / f"augmentations_{prefix}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
