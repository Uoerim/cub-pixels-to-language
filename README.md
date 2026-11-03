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

## Environment setup

Use Python 3.12 and run commands from the repository root.

```bash
python -m venv .venv
```

Activate the environment in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the CPU dependencies:

```bash
python -m pip install -r requirements.txt
```

For the NVIDIA CUDA 12.8 environment used in the measured experiments, install the framework builds before the other dependencies:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

Allow approximately 5 GB for dataset caches, plus space for the environment and checkpoints. The first training run also downloads the pretrained ResNet weights. To keep caches inside the project in PowerShell:

```powershell
$env:HF_HOME = "$PWD/.cache/huggingface"
$env:TORCH_HOME = "$PWD/.cache/torch"
$env:MPLCONFIGDIR = "$PWD/outputs/.matplotlib"
```

## Milestone 1 training

Audit the official CUB training/test splits first. This creates the cleaning manifest required by the training configurations.

```bash
python -m src.cli eda --config configs/eda.yaml
python -m src.cli train --task classification --config configs/cls_scratch.yaml
python -m src.cli train --task classification --config configs/cls_transfer.yaml
python -m src.cli tune --task classification --config configs/cls_transfer.yaml --search-space configs/search_space.yaml --method both --trials 2 --epochs 2
python -m src.cli train --task classification --config configs/cls_transfer_tuned.yaml
```

The scratch and transfer defaults are 50 and 30 epochs. The documented baseline experiments used `--epochs 5`; the tuned configuration also uses five epochs. Use a fresh `save_dir` for each repeat so existing histories are preserved. For CPU training, append `--device cpu` to a `train` command.

## Trained models and inference

Download the [Milestone 1 model and experiment folder](https://drive.google.com/drive/folders/10pyKKj_dRPa58NYPlg7gMpOW_TFn5dON) and place its contents in a directory named `artifacts` at the repository root. It contains seven trained checkpoints, their SHA-256 manifest, an example image, evaluation results, and experiment exports. The selected model is `artifacts/models/transfer_tuned.pth`.

Run prediction on the supplied real bird image:

```bash
python -m src.cli predict --task classification --image artifacts/evaluation/example_bird.jpg --checkpoint artifacts/models/transfer_tuned.pth --device cpu
```

Replace `--image` with another bird photograph to classify it. Inference loads the saved class names and weights without downloading pretrained weights. The cloud checkpoints contain model weights, configuration, class names, and history; optimizer/scheduler states remain in the original training checkpoints.

After running EDA, evaluate the downloaded models on the complete official test set:

```bash
python -m src.cli evaluate --task classification --checkpoint artifacts/models/scratch.pth --device cpu
python -m src.cli evaluate --task classification --checkpoint artifacts/models/transfer_baseline.pth --device cpu
python -m src.cli evaluate --task classification --checkpoint artifacts/models/transfer_tuned.pth --device cpu
python -m src.cli gradcam --config configs/cls_transfer_tuned.yaml --checkpoint artifacts/models/transfer_tuned.pth --split test --num_samples 8 --device cpu
python -m src.cli compare --reports scratch=outputs/scratch/evaluation/test/metrics.json transfer=outputs/transfer/evaluation/test/metrics.json tuned=outputs/transfer_tuned/evaluation/test/metrics.json
python -m src.cli benchmark --metrics outputs/transfer_tuned/evaluation/test/metrics.json
```

## Experiment tracking and verification

```bash
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db --port 5000
python -m unittest discover -v
python -m pip check
```

The supplied production configurations require MLflow. Each run records hyperparameters, epoch metrics, training curves, run IDs, and checkpoint references. JSON/CSV logs are also saved locally. Set `mlflow_required: false` in a configuration to allow local logging when MLflow is unavailable.

Detailed methodology, measured results, metric explanations, limitations, and run IDs are in [docs/milestone1.md](docs/milestone1.md). Dataset findings and preprocessing decisions are in [docs/data_findings.md](docs/data_findings.md).
