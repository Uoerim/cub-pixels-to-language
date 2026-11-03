import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_metrics(labels, logits, num_classes=None):
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    if logits.ndim != 2 or labels.ndim != 1 or len(labels) != len(logits) or not len(labels):
        raise ValueError("Provide nonempty labels [N] and logits [N, classes].")
    classes = logits.shape[1] if num_classes is None else num_classes
    if classes != logits.shape[1] or classes < 1 or not np.isfinite(logits).all():
        raise ValueError("Logits must be finite and match num_classes.")
    if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() >= classes:
        raise ValueError("Labels must be integer class indices in range.")
    predictions = logits.argmax(axis=1)
    top5 = np.argsort(-logits, axis=1, kind="stable")[:, :min(5, classes)]
    result = {"samples": len(labels), "num_classes": classes, "top1": float(100 * np.mean(predictions == labels)), "top5": float(100 * np.mean(np.any(top5 == labels[:, None], axis=1)))}
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, labels=np.arange(classes), average=average, zero_division=0)
        result.update({f"{average}_precision": float(100 * precision), f"{average}_recall": float(100 * recall), f"{average}_f1": float(100 * f1)})
    result["confusion_matrix"] = confusion_matrix(labels, predictions, labels=np.arange(classes)).tolist()
    return result


def evaluate_model(model, loader):
    device = next(model.parameters()).device
    previous = model.training
    labels, logits = [], []
    loss_sum = 0.0
    try:
        model.eval()
        with torch.inference_mode():
            for images, targets in loader:
                scores = model(images.to(device))
                loss_sum += torch.nn.functional.cross_entropy(scores, targets.to(device), reduction="sum").item()
                labels.extend(targets.cpu().tolist())
                logits.append(scores.cpu().numpy())
        if not logits:
            raise ValueError("Evaluation loader is empty.")
        scores = np.concatenate(logits)
        result = classification_metrics(labels, scores)
        result["loss"] = loss_sum / len(labels)
        return result, np.asarray(labels), scores.argmax(axis=1)
    finally:
        model.train(previous)


def dataset_signature(dataset):
    digest = hashlib.sha256()
    for index in dataset.indices:
        row = dataset.records[index]
        value = [int(index), int(row["label"]), row.get("id"), row.get("file_name")]
        digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def evaluate_command(args):
    from src.cli import load_config, load_model_checkpoint
    from src.data import build_dataloaders
    from src.training import seed_everything
    from src.visualization.evaluation import save_confusion_matrix

    supplied = load_config(args.config) if args.config else None
    model, config, names = load_model_checkpoint(args.checkpoint, supplied, args.device)
    seed_everything(config.get("seed", 42))
    loader = build_dataloaders(config)[args.split]
    current_names = loader.dataset.class_names
    if names and current_names and names != current_names:
        raise ValueError("Evaluation label names differ from the checkpoint class mapping.")
    result, labels, predictions = evaluate_model(model, loader)
    output = Path(args.output or str(Path(config.get("save_dir", "outputs")) / "evaluation" / args.split))
    output.mkdir(parents=True, exist_ok=True)
    matrix = result.pop("confusion_matrix")
    np.save(output / "confusion_matrix.npy", np.asarray(matrix))
    save_confusion_matrix(matrix, names or current_names, output / "confusion_matrix.png")
    provenance = {
        "split": args.split, "signature": dataset_signature(loader.dataset), "dataset_fingerprint": getattr(loader.dataset.records, "_fingerprint", None),
        "dataset": "Multimodal-Fatima/CUB_test" if args.split == "test" else "Multimodal-Fatima/CUB_train",
        "official_source": not bool(config.get("dataset_dir")), "img_size": config.get("img_size", 224),
        "checkpoint": str(Path(args.checkpoint).resolve()), "model_type": config.get("model_type", "scratch"),
        "model_kwargs": config.get("model_kwargs", {}), "seed": config.get("seed", 42),
    }
    report = {"metrics": result, "provenance": provenance, "class_names": names or current_names}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "predictions.json").write_text(json.dumps({"labels": labels.tolist(), "predictions": predictions.tolist()}), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def compare_reports(reports, output):
    if len(reports) < 2:
        raise ValueError("Compare at least two models.")
    signatures = {(row["provenance"]["split"], row["provenance"]["signature"], row["provenance"]["dataset_fingerprint"], row["metrics"]["samples"]) for row in reports}
    if len(signatures) != 1:
        raise ValueError("Model comparisons require identical split membership and dataset version.")
    from src.visualization.comparison import save_model_comparison

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    save_model_comparison(reports, output / "comparison.png")
    return reports


def compare_command(args):
    reports = []
    for entry in args.reports:
        name, separator, path = entry.partition("=")
        if not separator:
            raise ValueError("Use name=path/to/metrics.json for each report.")
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        reports.append({"name": name, **report})
    return compare_reports(reports, args.output)


def benchmark_report(report, reference):
    provenance, metrics = report["provenance"], report["metrics"]
    if provenance["split"] != "test" or not provenance["official_source"] or metrics["samples"] != 5794 or metrics["num_classes"] != 200:
        raise ValueError("Benchmarking requires the complete official 5,794-image, 200-class test split.")
    return {"reference": reference, "our_top1": metrics["top1"], "gap_percentage_points": metrics["top1"] - reference["top1"], "our_provenance": provenance,
            "interpretation": "Same official test split and per-image top-1 metric. Training budget, validation holdout, architecture, input resolution, pooling and test-time augmentation differ; this is a published-result comparison, not a reproduction. Only top-1 was reported by this reference. ImageNet pretraining may overlap CUB images, as documented by the dataset authors."}


def benchmark_command(args):
    report = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    result = benchmark_report(report, reference)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def register_commands(subparsers):
    evaluate = subparsers.add_parser("evaluate", help="Evaluate all images in the chosen split")
    evaluate.add_argument("--task", choices=["classification"], default="classification")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--config")
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--output")
    evaluate.add_argument("--device")
    evaluate.set_defaults(handler=evaluate_command)
    compare = subparsers.add_parser("compare", help="Compare scratch, transferred and tuned evaluation reports")
    compare.add_argument("--reports", nargs="+", required=True)
    compare.add_argument("--output", default="outputs/comparison")
    compare.set_defaults(handler=compare_command)
    benchmark = subparsers.add_parser("benchmark", help="Compare official test accuracy with a published result")
    benchmark.add_argument("--metrics", required=True)
    benchmark.add_argument("--reference", default="configs/benchmark.json")
    benchmark.add_argument("--output", default="outputs/benchmark.json")
    benchmark.set_defaults(handler=benchmark_command)
