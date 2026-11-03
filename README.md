# From Pixels to Language: Milestone 1

CSE 485 / CSE 488 classification and encoder training on CUB-200-2011. The final implementation lives in Python modules; notebooks are optional exploratory material.

## Setup

Use Python 3.12 in a fresh environment. Run commands from the repository root.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on Linux. For an NVIDIA GPU, install the matching framework builds first:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

For CPU execution, install only `requirements.txt` and pass `--device cpu` to training or prediction. CUDA installation follows the [official framework instructions](https://pytorch.org/get-started/locally/). First use downloads the dataset and pretrained weights. Allow approximately 5 GB for dataset caches and more for environments, experiment artifacts and checkpoints.

To keep caches inside the project in PowerShell:

```powershell
$env:HF_HOME = "$PWD/.cache/huggingface"
$env:TORCH_HOME = "$PWD/.cache/torch"
$env:MPLCONFIGDIR = "$PWD/outputs/.matplotlib"
```

Linux equivalents use `export HF_HOME="$PWD/.cache/huggingface"` and the corresponding variables above. These variables affect only the current shell.

## Data audit and preprocessing

```bash
python -m src.cli eda --config configs/eda.yaml
```

This loads exactly `Multimodal-Fatima/CUB_train` (`train`, 5,994 images) and `Multimodal-Fatima/CUB_test` (`test`, 5,794 images). Configurations pin both dataset revisions to those audited. EDA exports per-split JSON, image-statistics CSV, cleaned caption JSONL, sample images, distribution plots and `outputs/eda/cleaning_manifest.json`.

Each description is split on line boundaries; HTML entities are decoded; Unicode is normalized with NFKC; text is lowercased; punctuation/underscores become spaces; whitespace is collapsed; empty lines are discarded. Caption order and repeated nonempty captions are retained. Missing/non-string fields, unexpected caption counts, repeated captions, very short captions, digits and malformed characters are flagged. Rare words are exported for manual typo review; spelling is not changed automatically because uncommon species/anatomy terms may be valid. Raw records remain available for comparison.

The audit measures class counts/imbalance, width/height/aspect ratios, original image modes, pixel-weighted RGB mean/std, caption lengths/counts and vocabulary. It detects duplicate identifiers, filenames and decoded pixel hashes, conflicting duplicate labels, unreadable records and exact train/test overlap. Training configurations require the fingerprint-matched cleaning manifest: unreadable training records, duplicate training pixels and training images matching test pixels are excluded. Conflicting labels or unreadable test images stop training for review. The official test membership is preserved. No automatic rebalancing is applied; inspect class counts before changing the loss or sampler.

Images receive EXIF orientation correction and RGB conversion. Training uses 65–100% area crops with moderate aspect variation, horizontal flips, mild color jitter and rotations up to 10 degrees. These retain bird structure and species color cues while varying pose/background. Validation/test use deterministic resize/center crop. Both models use the same 224-pixel inputs and ImageNet normalization, which matches the selected [ResNet-18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

Validation is a seeded, stratified 15% holdout of the cleaned training split. The test set is not used for hyperparameter selection. Dataset versions and split signatures are recorded with evaluation results. Optional `dataset_dir` supports an offline dataset saved with `save_to_disk` under `train/` and `test/`; evaluations of this source are excluded from published benchmarking.

## Training and tuning

```bash
python -m src.cli train --task classification --config configs/cls_scratch.yaml
python -m src.cli train --task classification --config configs/cls_transfer.yaml
python -m src.cli tune --task classification --config configs/cls_transfer.yaml --search-space configs/search_space.yaml --method both --trials 4 --epochs 5
```

The existing scratch CNN has five Conv/BatchNorm/ReLU/Pool stages, adaptive pooling and a dropout/linear head. The transfer encoder supports `resnet18` (default), `resnet34`, and `resnet50`, initialized with fixed `IMAGENET1K_V1` weights. Set `model_kwargs.freeze_backbone: true` for a frozen feature extractor; its BatchNorm statistics remain frozen. Both expose `forward_features`, `forward_head`, `feature_dim`, and `get_last_conv_layer` through `BaseEncoder`. The trainer remains model-independent.

AdamW and cosine learning-rate decay are used. The original configurations retain 50 scratch epochs and 30 transfer epochs. `--epochs` provides an explicit shorter budget. Each run saves atomic best/latest checkpoints, configuration, per-epoch loss/top-1/top-5 logs and curves. Best checkpoints are selected by validation top-1. Use a fresh `save_dir` for each new training or tuning experiment; existing histories are protected from overwrite.

Grid search enumerates a Cartesian space; random search samples discrete combinations without replacement using the same seed. Both use the same trial/epoch budget, split and initialization seed. The supplied subset covers learning rate, weight decay and dropout. Supported search keys also include batch size. Truncated grid search follows the deterministic enumeration order, so it may cover only part of a larger space; use at least the full Cartesian count for an exhaustive grid. `--trials 8` covers the entire supplied space.

Search outputs under `outputs/transfer/tuning/` include isolated trial checkpoints/logs, `trials.json`, best trial configuration, `comparison.json`, `comparison.csv`, and a chart comparing best validation accuracy, trial count and wall time. Timing includes model/data-loader initialization, training, tracking and checkpoint writing, excluding the initial shared dataset download. `best_trial.yaml` identifies an already-trained trial; use its checkpoint rather than retraining into the same directory.

MLflow is required in the production configurations. It records hyperparameters, epoch metrics, curve artifacts, run IDs and checkpoint references with SHA-256 hashes. Tracking failure stops production runs; optional `mlflow_required: false` retains the original local-log fallback. Checkpoints stay local and are not uploaded by these commands.

```bash
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db --port 5000
```

Open the local tracking page and select the `cub-classification` experiment. Keep checkpoints together with their run configuration and hashes when handing them to the team leader.

## Evaluation, inference and explanations

```bash
python -m src.cli predict --task classification --image path/to/bird.jpg --checkpoint outputs/transfer/checkpoints/best_transfer.pth
python -m src.cli evaluate --task classification --checkpoint outputs/scratch/checkpoints/best_scratch.pth
python -m src.cli evaluate --task classification --checkpoint outputs/transfer/checkpoints/best_transfer.pth
python -m src.cli gradcam --config configs/cls_transfer.yaml --checkpoint outputs/transfer/checkpoints/best_transfer.pth --split test --num_samples 8
```

Prediction loads architecture and class names from the checkpoint, returns top-k probabilities, and never downloads pretrained weights. Evaluation processes the complete chosen split, independently of training batch limits. It saves metrics, predictions, a numeric confusion matrix and a figure under the run's `evaluation/test/` directory. All percentage metrics range from 0 to 100:

| Metric | Meaning |
| --- | --- |
| Top-1 / Top-5 | Fraction whose true species is first / among the five highest scores |
| Macro precision / recall / F1 | Equal-weight mean over all configured species; exposes weak classes |
| Weighted precision / recall / F1 | Class-support-weighted mean; reflects the observed class mix |
| Confusion matrix | Rows are true species, columns predicted species; off-diagonal counts are errors |
| Cross-entropy | Mean negative log probability of the true class; lower is better |

Undefined precision/recall/F1 values are set to zero. Top-5 uses all available classes in small smoke-test datasets. Grad-CAM targets the predicted class at the last convolution, overlays its gradient-weighted activation on the image, and saves a manifest with true/predicted labels and confidence. It seeks both correct and incorrect examples; if either category is absent, the manifest reflects the examples actually found. Inspect whether attention falls on the bird, its discriminative parts, or background shortcuts.

For the best tuning trial, substitute its exact checkpoint path from `comparison.json`:

```bash
python -m src.cli evaluate --checkpoint outputs/transfer/tuning/grid/trial_000/checkpoints/best_transfer.pth
python -m src.cli compare --reports scratch=outputs/scratch/evaluation/test/metrics.json transfer=outputs/transfer/evaluation/test/metrics.json tuned=outputs/transfer/tuning/grid/trial_000/evaluation/test/metrics.json
python -m src.cli benchmark --metrics outputs/transfer/evaluation/test/metrics.json
```

`trial_000` is an example path, not a claim that it won. Compare rejects mismatched split signatures or dataset versions. Use validation results to select the final model before opening test results. The published reference in `configs/benchmark.json` is the fine-tuned VGG-M + VGG-D bilinear model's 84.1% per-image accuracy in [Lin et al., Table 2](https://arxiv.org/abs/1504.07889). Benchmarking requires all 5,794 official test images and 200 classes and compares only the shared top-1 metric. Architecture, pooling, resolution, validation holdout, training budget and test-time processing differ, so this is a contextual published-result comparison. The [dataset authors](https://www.vision.caltech.edu/datasets/cub_200_2011/) also document possible ImageNet/CUB image overlap.

## Verification and handoff

```bash
python -m unittest discover -v
python -m pip check
```

Tests use tiny local fixtures for portability, including subprocess `train`, `predict`, `evaluate`, and `gradcam` commands, actual MLflow persistence, transfer feature/freezing behavior, checkpoint round trips, text anomalies, duplicate/corruption handling, metric arithmetic and both search methods. Fixture scores are verification evidence only. Real experiment results must come from the official dataset commands above.

Repository: [Uoerim/cub-pixels-to-language](https://github.com/Uoerim/cub-pixels-to-language). Keep generated figures, logs and weights under ignored `outputs/`. The team leader handles the trained-model Drive link, team number and final submission archive. No archive or hosted checkpoint link is created by this implementation. See `docs/audit.md` for the baseline audit and task ownership.

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
