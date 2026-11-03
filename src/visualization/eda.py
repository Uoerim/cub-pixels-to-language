from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_eda_plots(report, widths, heights, ratios, lengths, caption_counts, samples, output_dir):
    output_dir = Path(output_dir)
    split = report["split"]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes[0, 0].bar(range(len(report["class_counts"])), report["class_counts"])
    axes[0, 0].set(xlabel="Species ID", ylabel="Images", title="Class distribution")
    axes[0, 1].scatter(widths, heights, s=5, alpha=0.2)
    axes[0, 1].set(xlabel="Width (pixels)", ylabel="Height (pixels)", title="Image resolution")
    for axis, values, title, xlabel in ((axes[0, 2], ratios, "Aspect ratios", "Width / height"), (axes[1, 0], lengths, "Clean caption lengths", "Words"), (axes[1, 1], caption_counts, "Captions per image", "Captions")):
        axis.hist(values, bins=30)
        axis.set(title=title, xlabel=xlabel, ylabel="Count")
    axes[1, 2].bar(["R", "G", "B"], report["rgb_mean"], yerr=report["rgb_std"], capsize=4)
    axes[1, 2].set(title="Pixel-weighted channel mean and std", ylabel="Intensity [0, 1]", ylim=(0, 1))
    figure.suptitle(f"CUB {split}: {report['valid_images']} decoded images; vocabulary {report['vocabulary_size']}")
    figure.tight_layout()
    figure.savefig(output_dir / f"statistics_{split}.png", dpi=160)
    plt.close(figure)
    if samples:
        figure, axes = plt.subplots(2, 4, figsize=(16, 9))
        for axis in axes.flat:
            axis.axis("off")
        for axis, (image, label, caption) in zip(axes.flat, samples):
            axis.imshow(image)
            axis.set_title(f"Species {label}", fontsize=10)
            axis.text(0.5, -0.03, caption, transform=axis.transAxes, ha="center", va="top", wrap=True, fontsize=8)
        figure.tight_layout(h_pad=4)
        figure.savefig(output_dir / f"samples_{split}.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
