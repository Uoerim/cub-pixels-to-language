import csv
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


def resolve_device(name="auto"):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable on this machine.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this machine.")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_config(config, prefix=""):
    values = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            values.update(flatten_config(value, name))
        else:
            values[name] = json.dumps(value) if isinstance(value, (list, tuple)) else str(value)
    return values


class ExperimentTracker:
    def __init__(self, config, log_dir):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow = None
        self.run_id = None
        self.history = []
        self.config_path = self.log_dir / "config.json"
        self.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def start(self):
        if not self.config.get("mlflow", True):
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(self.config.get("mlflow_tracking_uri", "sqlite:///outputs/mlflow.db"))
            experiment = self.config.get("mlflow_experiment", "cub-classification")
            existing = mlflow.get_experiment_by_name(experiment)
            if existing is None:
                artifact_dir = self.log_dir.parent / "mlartifacts"
                mlflow.create_experiment(experiment, artifact_location=artifact_dir.resolve().as_uri())
            mlflow.set_experiment(experiment)
            run = mlflow.start_run(
                run_name=self.config.get("run_name", self.config.get("model_type", "scratch")),
                nested=mlflow.active_run() is not None,
            )
            self.mlflow = mlflow
            self.run_id = run.info.run_id
            mlflow.log_params(flatten_config(self.config))
        except Exception as error:
            self._disable(error)

    def _disable(self, error):
        warnings.warn(f"MLflow unavailable; JSON/CSV logging remains active: {error}", RuntimeWarning)
        self.close("FAILED")

    def log_epoch(self, row):
        self.history.append(dict(row))
        history_path = self.log_dir / "history.json"
        history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        with (self.log_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)
        if self.mlflow is not None:
            try:
                self.mlflow.log_metrics({key: value for key, value in row.items() if key != "epoch"}, step=row["epoch"])
            except Exception as error:
                self._disable(error)

    def log_artifacts(self):
        if self.mlflow is not None:
            try:
                self.mlflow.log_artifacts(str(self.log_dir), artifact_path="training")
            except Exception as error:
                self._disable(error)

    def close(self, status="FINISHED"):
        if self.mlflow is not None:
            try:
                self.mlflow.end_run(status=status)
            except Exception:
                pass
            self.mlflow = None


class Trainer:
    def __init__(self, model, config, class_names=None, callbacks=None):
        self.config = dict(config)
        self.device = resolve_device(config.get("device", "auto"))
        self.model = model.to(self.device)
        self.class_names = list(class_names or [])
        self.callbacks = list(callbacks or [])
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.get("lr", 0.001),
            weight_decay=config.get("weight_decay", 0.0001),
        )
        epochs = config.get("epochs", 50)
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.save_dir = Path(config.get("save_dir", "outputs"))
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = config.get("model_type", "scratch")
        self.tracker = ExperimentTracker(config, self.save_dir / "logs" / self.model_type)
        self.history = self.tracker.history
        self.best_accuracy = -math.inf

    def _run_epoch(self, loader, training, max_batches=None):
        if max_batches is not None and max_batches < 1:
            raise ValueError("Batch limits must be positive.")
        self.model.train(training)
        total_loss = 0.0
        total_top1 = 0
        total_top5 = 0
        total_samples = 0
        progress = tqdm(loader, desc="Train" if training else "Val", leave=False, disable=not self.config.get("progress", True))
        with torch.set_grad_enabled(training):
            for batch_index, (images, labels) in enumerate(progress):
                if max_batches is not None and batch_index >= max_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss encountered.")
                if training:
                    loss.backward()
                    self.optimizer.step()
                predictions = logits.detach().topk(min(5, logits.shape[1]), dim=1).indices
                matches = predictions.eq(labels[:, None])
                count = labels.size(0)
                total_samples += count
                total_loss += loss.item() * count
                total_top1 += matches[:, 0].sum().item()
                total_top5 += matches.any(dim=1).sum().item()
                progress.set_postfix(loss=f"{total_loss / total_samples:.4f}", top1=f"{100 * total_top1 / total_samples:.2f}%")
        if total_samples == 0:
            raise ValueError("The data loader is empty.")
        return {
            "loss": total_loss / total_samples,
            "top1": 100 * total_top1 / total_samples,
            "top5": 100 * total_top5 / total_samples,
        }

    def evaluate(self, loader, max_batches=None):
        return self._run_epoch(loader, training=False, max_batches=max_batches)

    def _save_checkpoint(self, epoch, is_best):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_top1": self.best_accuracy,
            "config": self.config,
            "class_names": self.class_names,
            "history": self.history,
        }
        names = [f"latest_{self.model_type}.pth"]
        if is_best:
            names.append(f"best_{self.model_type}.pth")
        for name in names:
            path = self.checkpoint_dir / name
            temporary = path.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(path)

    def fit(self, train_loader, val_loader):
        self.tracker.start()
        status = "FAILED"
        try:
            for epoch in range(1, self.config.get("epochs", 50) + 1):
                learning_rate = self.optimizer.param_groups[0]["lr"]
                train_metrics = self._run_epoch(train_loader, True, self.config.get("max_train_batches"))
                val_metrics = self.evaluate(val_loader, self.config.get("max_val_batches"))
                row = {"epoch": epoch, "lr": learning_rate}
                row.update({f"train_{key}": value for key, value in train_metrics.items()})
                row.update({f"val_{key}": value for key, value in val_metrics.items()})
                self.tracker.log_epoch(row)
                self.scheduler.step()
                is_best = val_metrics["top1"] > self.best_accuracy
                if is_best:
                    self.best_accuracy = val_metrics["top1"]
                self._save_checkpoint(epoch, is_best)
                for callback in self.callbacks:
                    callback(self, row)
                print(f"Epoch {epoch}: train loss {row['train_loss']:.4f}, val loss {row['val_loss']:.4f}, val top-1 {row['val_top1']:.2f}%, top-5 {row['val_top5']:.2f}%")
            self.tracker.log_artifacts()
            status = "FINISHED"
            return self.history
        finally:
            self.tracker.close(status)
