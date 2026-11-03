import argparse
import json
from importlib import import_module
from pathlib import Path

import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration must contain a YAML mapping.")
    if config.get("task", "classification") != "classification":
        raise ValueError("This milestone supports the classification task.")
    return config


def load_model_checkpoint(path, config=None, device_name=None):
    import torch

    from src.models import build_model
    from src.training import resolve_device

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("Use a checkpoint saved by Trainer with model weights and configuration.")
    saved_config = checkpoint["config"]
    config = config or {}
    architecture_keys = ("model_type", "num_classes", "img_size", "dropout", "model_module", "model_class", "model_kwargs")
    for key in architecture_keys:
        if key in config and key in saved_config and config[key] != saved_config[key]:
            raise ValueError(f"Configuration '{key}' differs from the saved checkpoint.")
    effective_config = {**saved_config, **config}
    device = resolve_device(device_name or effective_config.get("device", "auto"))
    model = build_model(effective_config, for_inference=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, effective_config, list(checkpoint.get("class_names", []))


def train_command(args):
    import torch

    from src.data import build_dataloaders
    from src.models import build_model
    from src.training import Trainer, seed_everything
    from src.visualization.plots import save_augmented_samples, save_training_curves

    config = load_config(args.config)
    if config.get("num_threads"):
        torch.set_num_threads(config["num_threads"])
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["epochs"] = args.epochs
    seed_everything(config.get("seed", 42))
    model = build_model(config)
    loaders = build_dataloaders(config)
    names = loaders["train"].dataset.class_names
    figures = Path(config.get("save_dir", "outputs")) / "figures"
    save_augmented_samples(loaders["train"].dataset, figures, prefix=config.get("model_type", "scratch"))
    trainer = Trainer(model, config, class_names=names)
    history = trainer.fit(loaders["train"], loaders["val"])
    save_training_curves(history, figures, prefix=config.get("model_type", "scratch"))
    print(f"Best validation top-1: {trainer.best_accuracy:.2f}%")
    print(f"Checkpoints saved to {trainer.checkpoint_dir}")
    return history


def predict_command(args):
    import torch
    from PIL import Image, ImageOps

    from src.data import build_transforms

    if args.top_k < 1:
        raise ValueError("top_k must be positive.")
    config = load_config(args.config) if args.config else None
    model, config, names = load_model_checkpoint(args.checkpoint, config, args.device)
    _, transform = build_transforms(config.get("img_size", 224))
    with Image.open(args.image) as image:
        image = transform(ImageOps.exif_transpose(image).convert("RGB"))
    device = next(model.parameters()).device
    with torch.inference_mode():
        probabilities = model(image.unsqueeze(0).to(device)).softmax(dim=1)[0]
        scores, indices = probabilities.topk(min(args.top_k, probabilities.numel()))
    results = [
        {"class_id": int(index), "class_name": names[int(index)] if int(index) < len(names) else str(int(index)), "probability": float(score)}
        for score, index in zip(scores.cpu(), indices.cpu())
    ]
    print(json.dumps(results, indent=2))
    return results


def gradcam_command(args):
    from src.data import build_dataloaders
    from src.training import seed_everything
    from src.visualization.gradcam import save_gradcam_examples

    if args.num_samples < 1:
        raise ValueError("num_samples must be positive.")
    config = load_config(args.config)
    model, config, names = load_model_checkpoint(args.checkpoint, config, args.device)
    seed_everything(config.get("seed", 42))
    loaders = build_dataloaders(config)
    output_dir = Path(config.get("save_dir", "outputs")) / "gradcam"
    paths = save_gradcam_examples(model, loaders[args.split], output_dir, args.num_samples, names, args.max_scan_samples)
    print(f"Saved {len(paths)} Grad-CAM examples to {output_dir}")
    return paths


def register_extension_commands(subparsers):
    for module_name in ("src.data.eda", "src.training.tuning", "src.evaluation.metrics"):
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name or module_name.startswith(f"{error.name}."):
                continue
            raise
        register = getattr(module, "register_commands", None)
        if register is not None:
            register(subparsers)


def build_parser():
    parser = argparse.ArgumentParser(description="CUB classification training and explainability")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train a classification model")
    train.add_argument("--task", choices=["classification"], default="classification")
    train.add_argument("--config", required=True)
    train.add_argument("--device", default=None)
    train.add_argument("--epochs", type=int)
    train.set_defaults(handler=train_command)
    predict = subparsers.add_parser("predict", help="Predict a bird class from an image")
    predict.add_argument("--task", choices=["classification"], default="classification")
    predict.add_argument("--image", required=True)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--config")
    predict.add_argument("--device", default=None)
    predict.add_argument("--top_k", type=int, default=5)
    predict.set_defaults(handler=predict_command)
    gradcam = subparsers.add_parser("gradcam", help="Save correct and incorrect prediction explanations")
    gradcam.add_argument("--task", choices=["classification"], default="classification")
    gradcam.add_argument("--config", required=True)
    gradcam.add_argument("--checkpoint", required=True)
    gradcam.add_argument("--num_samples", type=int, default=8)
    gradcam.add_argument("--split", choices=["val", "test"], default="test")
    gradcam.add_argument("--max_scan_samples", type=int)
    gradcam.add_argument("--device", default=None)
    gradcam.set_defaults(handler=gradcam_command)
    register_extension_commands(subparsers)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
