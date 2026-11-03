import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

from src.cli import load_model_checkpoint, main
from src.evaluation.metrics import benchmark_report, classification_metrics, compare_reports
from src.models import build_model
from src.models.transfer_cnn import TransferCNN
from src.training.tuning import run_search, trial_parameters
from src.visualization.gradcam import GradCAM
from tests.test_pipeline import make_records


torch.set_num_threads(2)


class MilestoneTests(unittest.TestCase):
    def test_transfer_features_freezing_gradcam_and_checkpoint(self):
        config = {"model_type": "transfer", "num_classes": 2, "dropout": 0.1, "model_kwargs": {"pretrained": False, "freeze_backbone": True}}
        model = build_model(config)
        model.train()
        self.assertFalse(model.features.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.features.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))
        images = torch.randn(2, 3, 64, 64)
        model.eval()
        with torch.no_grad():
            features = model.forward_features(images)
            self.assertEqual(features.shape, (2, 512, 2, 2))
            torch.testing.assert_close(model(images), model.forward_head(features))
        with GradCAM(model, output_size=(64, 64)) as cam:
            maps, scores = cam(images)
            self.assertEqual(maps.shape, (2, 64, 64))
            self.assertTrue(torch.isfinite(maps).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            configurations = [{**config, "model_kwargs": {"pretrained": True, "freeze_backbone": True}}, {key: value for key, value in config.items() if key != "model_kwargs"}]
            for saved in configurations:
                torch.save({"config": saved, "model_state_dict": model.state_dict()}, path)
                with patch("torchvision.models.resnet18", wraps=__import__("torchvision").models.resnet18) as constructor:
                    restored, _, _ = load_model_checkpoint(path, device_name="cpu")
                    self.assertIsNone(constructor.call_args.kwargs["weights"])
                with torch.no_grad():
                    torch.testing.assert_close(model(images), restored(images))
        with self.assertRaisesRegex(ValueError, "Unsupported backbone"):
            TransferCNN(backbone="unknown", pretrained=False)

    def test_metrics_known_values_and_absent_classes(self):
        logits = np.array([[9, 1, 0], [8, 3, 0], [1, 9, 0], [1, 8, 0]], dtype=float)
        result = classification_metrics(np.array([0, 1, 1, 2]), logits)
        self.assertEqual(result["top1"], 50)
        self.assertEqual(result["top5"], 100)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [1, 1, 0], [0, 1, 0]])
        self.assertAlmostEqual(result["macro_precision"], 100 / 3)
        self.assertAlmostEqual(result["macro_recall"], 50)
        self.assertAlmostEqual(result["weighted_recall"], 50)
        with self.assertRaises(ValueError):
            classification_metrics([], [])
        with self.assertRaises(ValueError):
            classification_metrics([0], [[float("nan"), 1]])

    def test_search_reproducibility_and_validation_only_trials(self):
        space = {"lr": [0.001, 0.002], "dropout": [0.1, 0.2]}
        self.assertEqual(trial_parameters(space, "random", 3, 42), trial_parameters(space, "random", 3, 42))
        self.assertEqual(len(trial_parameters(space, "random", 20, 42)), 4)
        with self.assertRaises(ValueError):
            trial_parameters({"val_split": [0.2]}, "grid", 2, 42)
        with tempfile.TemporaryDirectory() as directory:
            config = {"model_type": "scratch", "num_classes": 2, "img_size": 64, "epochs": 1, "batch_size": 4, "val_split": 0.25, "save_dir": directory, "device": "cpu", "mlflow": False, "progress": False, "max_train_batches": 1, "max_val_batches": 1}
            result = run_search(config, space, "both", 2, make_records(), make_records(4))
            self.assertFalse(result["test_used_for_selection"])
            self.assertEqual([row["trials"] for row in result["methods"]], [2, 2])
            for row in result["methods"]:
                self.assertTrue(Path(row["best_checkpoint"]).exists())
                self.assertGreater(row["total_seconds"], 0)
            self.assertEqual(len(json.loads((Path(directory) / "tuning" / "trials.json").read_text())), 4)
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_search(config, space, "both", 2, make_records(), make_records(4))

    def test_cli_subprocess_train_predict_evaluate_and_gradcam(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            make_records().save_to_disk(str(data / "train"))
            make_records(4).save_to_disk(str(data / "test"))
            image = root / "bird.png"
            Image.new("RGB", (80, 80), "brown").save(image)
            config = {"task": "classification", "model_type": "scratch", "num_classes": 2, "img_size": 64, "epochs": 1, "batch_size": 4, "val_split": 0.25, "dataset_dir": str(data), "save_dir": str(root / "run"), "device": "cpu", "mlflow": False, "progress": False, "max_train_batches": 1, "max_val_batches": 1, "num_threads": 2}
            path = root / "config.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            checkpoint = root / "run" / "checkpoints" / "best_scratch.pth"
            commands = [
                ["train", "--task", "classification", "--config", str(path)],
                ["predict", "--task", "classification", "--checkpoint", str(checkpoint), "--image", str(image)],
                ["evaluate", "--checkpoint", str(checkpoint)],
                ["gradcam", "--config", str(path), "--checkpoint", str(checkpoint), "--num_samples", "2"],
            ]
            for args in commands:
                result = subprocess.run([sys.executable, "-m", "src.cli", *args], capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "run" / "evaluation" / "test" / "metrics.json").read_text())
            self.assertEqual(report["metrics"]["samples"], 4)
            self.assertFalse(report["provenance"]["official_source"])
            with self.assertRaisesRegex(ValueError, "complete official"):
                benchmark_report(report, {"top1": 84.1})
            with contextlib.redirect_stdout(io.StringIO()):
                main(["compare", "--reports", f"scratch={root / 'run' / 'evaluation' / 'test' / 'metrics.json'}", f"copy={root / 'run' / 'evaluation' / 'test' / 'metrics.json'}", "--output", str(root / "comparison")])
            report["name"] = "changed"
            changed = {**report, "provenance": {**report["provenance"], "signature": "different"}}
            with self.assertRaisesRegex(ValueError, "identical split"):
                compare_reports([report, changed], root / "invalid")
