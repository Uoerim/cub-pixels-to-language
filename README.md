## Yosuf's Tasks

**Milestone 1:**

- Project structure and CLI entry point (`src/cli.py`, `configs/`).
- PyTorch CUB dataset wrapper and image augmentations (`src/data/dataset.py`).
- Base encoder interface and custom CNN from scratch (`src/models/base_encoder.py`, `src/models/scratch_cnn.py`).
- Core training loop and MLflow tracking (`src/training/trainer.py`).
- Grad-CAM explainability and training/augmentation plots (`src/visualization/gradcam.py`, `src/visualization/plots.py`).

**Milestone 2:** Task 2.2 Image-to-Text Captioning: freezing the visual encoder, vocabulary/tokenizer, sequence decoder, and BLEU/ROUGE-L evaluation.

## Helmy's Tasks

**Milestone 1:**

- Exploratory data analysis and caption text preprocessing (`notebooks/eda.ipynb`, `src/data/preprocess.py`).
- Transfer learning CNN model (`src/models/transfer_cnn.py`).
- Hyperparameter tuning with Grid Search and Random Search (`src/training/tuning.py`).
- Evaluation metrics and confusion matrix (`src/evaluation/metrics.py`).
- Model checkpoint uploads and final archive packaging.

**Milestone 2:** Task 2.1 Image Colorization: grayscale-to-RGB pairs, encoder-decoder/U-Net architecture, and PSNR/MAE evaluation.
