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

Images receive EXIF orientation correction and RGB conversion. Training uses 65â€“100% area crops with moderate aspect variation, horizontal flips, mild color jitter and rotations up to 10 degrees. These retain bird structure and species color cues while varying pose/background. Validation/test use deterministic resize/center crop. Both models use the same 224-pixel inputs and ImageNet normalization, which matches the selected [ResNet-18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

Validation is a seeded, stratified 15% holdout of the cleaned training split. The test set is not used for hyperparameter selection. Dataset versions and split signatures are recorded with evaluation results. Optional `dataset_dir` supports an offline dataset saved with `save_to_disk` under `train/` and `test/`; evaluations of this source are excluded from published benchmarking.

## Training and tuning

```bash
python -m src.cli train --task classification --config configs/cls_scratch.yaml
python -m src.cli train --task classification --config configs/cls_transfer.yaml
python -m src.cli tune --task classification --config configs/cls_transfer.yaml --search-space configs/search_space.yaml --method both --trials 2 --epochs 2
```

The existing scratch CNN has five Conv/BatchNorm/ReLU/Pool stages, adaptive pooling and a dropout/linear head. The transfer encoder supports `resnet18` (default), `resnet34`, and `resnet50`, initialized with fixed `IMAGENET1K_V1` weights. Set `model_kwargs.freeze_backbone: true` for a frozen feature extractor; its BatchNorm statistics remain frozen. Both expose `forward_features`, `forward_head`, `feature_dim`, and `get_last_conv_layer` through `BaseEncoder`. The trainer remains model-independent.

AdamW and cosine learning-rate decay are used. The original configurations retain 50 scratch epochs and 30 transfer epochs. `--epochs` provides an explicit shorter budget. Each run saves atomic best/latest checkpoints, configuration, per-epoch loss/top-1/top-5 logs and curves. Best checkpoints are selected by validation top-1. Use a fresh `save_dir` for each new training or tuning experiment; existing histories are protected from overwrite.

Grid search enumerates a Cartesian space; random search samples discrete combinations without replacement using the same seed. Both use the same trial/epoch budget, split and initialization seed. The supplied space compares two learning rates with fixed weight decay and dropout. Supported search keys also include batch size. Both methods cover this entire space with two trials. When expanding the space, use at least its full Cartesian count for an exhaustive grid; a smaller trial budget truncates the grid in deterministic enumeration order.

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

Repository: [Uoerim/cub-pixels-to-language](https://github.com/Uoerim/cub-pixels-to-language). Keep generated figures, logs and weights under ignored `outputs/`. The team leader handles the trained-model Drive link, team number and final submission archive. The repository README retains its original task ownership sections. Full-data measurements and cleaning decisions are documented in [data_findings.md](data_findings.md).

## Measured runs: 2026-11-03

These results use all 5,094 cleaned training images, the same 899-image stratified validation holdout and all 5,794 official test images. Hardware was a 12 GB RTX 3060 on Windows, with Python 3.12.14, PyTorch 2.7.1 + CUDA 12.8 and Torchvision 0.22.1. No training or validation batch limits were applied. Each baseline and the final tuned model trained for five epochs. The longer 50/30-epoch defaults were not run to completion in this experiment.

The measured sequence, after EDA, is:

```bash
python -m src.cli train --task classification --config configs/cls_scratch.yaml --epochs 5
python -m src.cli train --task classification --config configs/cls_transfer.yaml --epochs 5
python -m src.cli tune --task classification --config configs/cls_transfer.yaml --search-space configs/search_space.yaml --method both --trials 2 --epochs 2
python -m src.cli train --task classification --config configs/cls_transfer_tuned.yaml
```

The final configuration is a fresh ImageNet-initialized ResNet-18 run with the selected learning rate `0.0003`, rather than an extension of a two-epoch trial. Dropout is 0.5 and weight decay is 0.0001 throughout this comparison. The final checkpoint was selected by best validation top-1 before any test-set model evaluation.

| Model | Best validation top-1 | Test top-1 | Test top-5 | Test macro F1 | Test weighted F1 | Test loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scratch CNN | 4.67% | 4.09% | 16.29% | 2.49% | 2.50% | 4.6963 |
| Transfer baseline | 54.06% | 58.94% | 86.31% | 57.10% | 57.03% | 2.3801 |
| Tuned transfer | 67.30% | 69.57% | 91.75% | 68.94% | 68.92% | 1.3509 |

| Model | Macro precision | Macro recall | Weighted precision | Weighted recall |
| --- | ---: | ---: | ---: | ---: |
| Scratch CNN | 3.67% | 4.18% | 3.73% | 4.09% |
| Transfer baseline | 60.66% | 59.22% | 60.81% | 58.94% |
| Tuned transfer | 69.78% | 69.72% | 69.87% | 69.57% |

The tuned model improves test top-1 by 10.63 percentage points over the transfer baseline under the same five-epoch budget. The scratch encoder remains strongly underfit, so this experiment establishes a short-budget baseline rather than its converged performance. Training curves are saved with each run. Validation curves can exceed online training accuracy because training uses augmentation/dropout and aggregates predictions while the parameters are changing.

| Search method | Trials | Epochs per trial | Best validation top-1 | Total seconds |
| --- | ---: | ---: | ---: | ---: |
| Grid | 2 | 2 | 48.50% | 323.38 |
| Random | 2 | 2 | 48.94% | 252.96 |

Both methods exhaust the same two-learning-rate space and choose `0.0003`. These are single-machine observations, with GPU determinism not enforced and possible cache/background-load effects. The small score/time differences do not establish a general advantage for either method. Larger spaces and repeated runs would be needed for that conclusion.

The selected model's 69.57% test top-1 is 14.53 percentage points below the configured published 84.1% bilinear reference. The shared test split and per-image metric support the comparison; different architectures, training budgets, input processing and the held-out validation data limit its interpretation. This is not a reproduction of the published model.

Seven real MLflow runs completed with status `FINISHED`:

| Run | Run ID |
| --- | --- |
| Scratch | `e837ed1695f9467ba1aa04a56f32b5f3` |
| Transfer baseline | `87059a1136994614857a15a25709fbd0` |
| Grid trial 0 | `69a6d083f5454f22b08f701103fbee53` |
| Grid trial 1 | `b8fd1eaf187e48eaa913f6bd3c0d7070` |
| Random trial 0 | `cd10c31693724e38af97c3c1babeb6ef` |
| Random trial 1 | `d1c5857cc9f34b5ba6af76d3fdab0b9e` |
| Tuned transfer | `6decff962f70426c959471f7d1a73d1b` |

Each run has configuration, epoch metrics, training curves and hashed checkpoint references. The scratch run initially used two data-loader workers; subsequent runs use zero to avoid Windows memory pressure. Its checkpoint retains the actual settings. Seeds and dataset revisions are recorded, but worker RNG streams and GPU arithmetic mean reruns need not reproduce scores bit for bit.

Final-model evaluation and explanations were verified with:

```bash
python -m src.cli evaluate --checkpoint outputs/transfer_tuned/checkpoints/best_transfer.pth
python -m src.cli gradcam --config configs/cls_transfer_tuned.yaml --checkpoint outputs/transfer_tuned/checkpoints/best_transfer.pth --split test --num_samples 4
python -m src.cli compare --reports scratch=outputs/scratch/evaluation/test/metrics.json transfer=outputs/transfer/evaluation/test/metrics.json tuned=outputs/transfer_tuned/evaluation/test/metrics.json
python -m src.cli benchmark --metrics outputs/transfer_tuned/evaluation/test/metrics.json
```

Grad-CAM saved two correct and two incorrect cases after scanning nine images. In `correct_00000_true_0_pred_0.png`, the strongest response covers the bird's torso/wing region. In `wrong_00003_true_0_pred_2.png`, the response covers the bird's head and body, yet the model confuses Black footed Albatross with Sooty Albatross. Focusing on the bird is therefore insufficient to resolve every fine-grained species distinction. The CLI also classified an exported real test image successfully using the selected checkpoint.

The final full verification completed with 21 passing tests. Generated distributions, curves, confusion matrices, comparisons, predictions, Grad-CAM images, run logs and checkpoint files remain local under `outputs/`; they are reproducible through the commands above and are not included in the source pull request.

