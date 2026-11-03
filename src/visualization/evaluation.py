import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(matrix, class_names, path):
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    axis.set(xlabel="Predicted species", ylabel="True species", title="Confusion matrix (counts)")
    if len(matrix) <= 20:
        names = class_names or [str(value) for value in range(len(matrix))]
        axis.set_xticks(range(len(matrix)), names, rotation=90)
        axis.set_yticks(range(len(matrix)), names)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
