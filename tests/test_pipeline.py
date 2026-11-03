import ast
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import tokenize
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import torch
import yaml
from datasets import ClassLabel, Dataset, Features, Image as DatasetImage, Value
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.cli import build_parser, load_model_checkpoint, main
from src.data import CUBDataset, build_dataloaders, build_transforms
from src.models import ScratchCNN, build_model, register_model
from src.training import Trainer, seed_everything
from src.training.trainer import ExperimentTracker
from src.visualization.gradcam import GradCAM, overlay_heatmap, save_gradcam_examples


torch.set_num_threads(2)


def make_records(count=16):
    features = Features({"image": DatasetImage(), "label": ClassLabel(names=["bird one", "bird two"]), "description": Value("string")})
    images = [Image.new("L" if index % 2 == 0 else "RGB", (80, 72), index * 12) for index in range(count)]
    return Dataset.from_dict({"image": images, "label": [index % 2 for index in range(count)], "description": ["A Blue Bird!\nIts beak is long." for _ in range(count)]}, features=features)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        Path("outputs").mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir="outputs")
        self.output = Path(self.temporary.name)
        self.config = {"task": "classification", "model_type": "scratch", "num_classes": 2, "img_size": 64, "batch_size": 4, "val_split": 0.25, "seed": 42, "num_workers": 0, "epochs": 2, "lr": 0.001, "weight_decay": 0.0001, "dropout": 0.2, "save_dir": str(self.output), "device": "cpu", "mlflow": False, "progress": False}

    def tearDown(self):
        self.temporary.cleanup()

    def loaders(self):
        return build_dataloaders(self.config, make_records(), make_records(4))

    def test_dataset_rgb_captions_and_normalization(self):
        train, test = build_transforms()
        dataset = CUBDataset(make_records(), test)
        image, label = dataset[0]
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertEqual(label, 0)
        self.assertTrue(torch.isfinite(image).all())
        self.assertEqual(dataset.captions(0), ["a blue bird", "its beak is long"])
        self.assertEqual(dataset.preprocess_captions("\n BLUE-gray_bird?! \n\n"), ["blue gray bird"])
        self.assertEqual(train(Image.new("L", (240, 250)).convert("RGB")).shape, (3, 224, 224))
        overlay = overlay_heatmap(image, np.zeros((224, 224), dtype=np.float32))
        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertTrue(np.isfinite(overlay).all())

    def test_stratified_split_is_reproducible_and_disjoint(self):
        first, second = self.loaders(), self.loaders()
        train, val = first["train"].dataset, first["val"].dataset
        self.assertFalse(set(train.indices) & set(val.indices))
        self.assertEqual(set(train.indices) | set(val.indices), set(range(16)))
        self.assertEqual(train.indices, second["train"].dataset.indices)
        self.assertEqual({val[index][1] for index in range(len(val))}, {0, 1})
        self.assertEqual(len(first["test"].dataset), 4)
        torch.testing.assert_close(val[0][0], val[0][0])

    def test_huggingface_loader_uses_official_splits(self):
        with patch("src.data.dataset.load_dataset", side_effect=[make_records(), make_records(4)]) as load:
            build_dataloaders(self.config)
        self.assertEqual(load.call_args_list[0].args, ("Multimodal-Fatima/CUB_train",))
        self.assertEqual(load.call_args_list[0].kwargs["split"], "train")
        self.assertEqual(load.call_args_list[1].args, ("Multimodal-Fatima/CUB_test",))
        self.assertEqual(load.call_args_list[1].kwargs["split"], "test")

    def test_encoder_features_and_classifier_are_separate(self):
        model = ScratchCNN().eval()
        images = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            features = model.forward_features(images)
            logits = model(images)
            torch.testing.assert_close(logits, model.forward_head(features))
        self.assertEqual(features.shape, (2, 512, 7, 7))
        self.assertEqual(logits.shape, (2, 200))
        self.assertEqual(sum(isinstance(layer, nn.Conv2d) for layer in model.modules()), 5)
        self.assertIsInstance(model.get_last_conv_layer(), nn.Conv2d)

    def test_trainer_checkpoints_logs_and_predict(self):
        loaders = self.loaders()
        trainer = Trainer(build_model(self.config), self.config, ["bird one", "bird two"])
        initial = trainer.model.classifier[-1].weight.detach().clone()
        history = trainer.fit(loaders["train"], loaders["val"])
        self.assertEqual(len(history), 2)
        self.assertFalse(torch.equal(initial, trainer.model.classifier[-1].weight))
        for row in history:
            for key in ("train_loss", "val_loss", "train_top1", "val_top1", "train_top5", "val_top5", "lr"):
                self.assertTrue(math.isfinite(row[key]))
            self.assertEqual(row["train_top5"], 100)
        best = self.output / "checkpoints" / "best_scratch.pth"
        latest = self.output / "checkpoints" / "latest_scratch.pth"
        self.assertTrue(best.exists() and latest.exists())
        reloaded, _, names = load_model_checkpoint(latest, device_name="cpu")
        trainer.model.eval()
        images, _ = next(iter(loaders["val"]))
        with torch.no_grad():
            torch.testing.assert_close(reloaded(images), trainer.model(images))
        self.assertEqual(names, ["bird one", "bird two"])
        log_dir = self.output / "logs" / "scratch"
        self.assertEqual(json.loads((log_dir / "history.json").read_text()), history)
        self.assertEqual(len((log_dir / "metrics.csv").read_text().splitlines()), 3)
        image_path = self.output / "bird.png"
        Image.new("L", (90, 80), 110).save(image_path)
        with contextlib.redirect_stdout(io.StringIO()):
            results = main(["predict", "--image", str(image_path), "--checkpoint", str(best), "--device", "cpu"])
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(sum(row["probability"] for row in results), 1, places=5)
        with self.assertRaises(ValueError):
            load_model_checkpoint(best, {"num_classes": 200})

    def test_accuracies_are_sample_weighted(self):
        logits = torch.tensor([[5., 0., 0., 0., 0., 0.], [0., 5., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 5.]])
        labels = torch.tensor([0, 1, 0])
        model = nn.Linear(6, 6, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(6))
        trainer = Trainer(model, self.config)
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        metrics = trainer.evaluate(loader)
        self.assertAlmostEqual(metrics["top1"], 200 / 3)
        self.assertAlmostEqual(metrics["loss"], nn.functional.cross_entropy(logits, labels).item(), places=6)

    def test_mlflow_failure_keeps_local_logs(self):
        tracker = ExperimentTracker({**self.config, "mlflow": True}, self.output / "fallback")
        with patch.dict(sys.modules, {"mlflow": None}):
            with self.assertWarns(RuntimeWarning):
                tracker.start()
        tracker.log_epoch({"epoch": 1, "val_loss": 1.0})
        self.assertTrue((self.output / "fallback" / "metrics.csv").exists())
        self.assertIsNone(tracker.mlflow)

    def test_mlflow_records_real_run(self):
        uri = "sqlite:///" + (self.output / "tracking.db").resolve().as_posix()
        config = {**self.config, "mlflow": True, "mlflow_tracking_uri": uri, "mlflow_experiment": "pipeline-check", "model_kwargs": {"option": 3}}
        code = "\n".join([
            "import json, sys, mlflow",
            "from pathlib import Path",
            "from src.training.trainer import ExperimentTracker",
            "config = json.loads(sys.argv[1])",
            "tracker = ExperimentTracker(config, Path(config['save_dir']) / 'tracked')",
            "tracker.start()",
            "assert tracker.mlflow is not None",
            "tracker.log_epoch({'epoch': 1, 'val_loss': 0.75, 'train_top1': 50.0, 'val_top5': 100.0})",
            "tracker.log_artifacts()",
            "tracker.close()",
            "run = mlflow.MlflowClient(tracking_uri=config['mlflow_tracking_uri']).get_run(tracker.run_id)",
            "assert run.info.status == 'FINISHED'",
            "assert run.data.params['model_kwargs.option'] == '3'",
            "assert run.data.metrics['val_loss'] == 0.75",
        ])
        result = subprocess.run([sys.executable, "-c", code, json.dumps(config)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_gradcam_with_frozen_backbone_and_hook_cleanup(self):
        model = ScratchCNN(num_classes=2)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        layer = model.get_last_conv_layer()
        with GradCAM(model) as gradcam:
            maps, logits = gradcam(torch.randn(2, 3, 64, 64), [0, 1])
            self.assertEqual(maps.shape, (2, 224, 224))
            self.assertEqual(logits.shape, (2, 2))
            self.assertTrue(torch.isfinite(maps).all())
            self.assertGreaterEqual(float(maps.min()), 0)
            self.assertLessEqual(float(maps.max()), 1)
            self.assertTrue(model.training)
        self.assertEqual(len(layer._forward_hooks), 0)
        self.assertEqual(len(layer._backward_hooks), 0)

    def test_gradcam_saves_correct_and_wrong_predictions(self):
        model = ScratchCNN(num_classes=2).eval()
        images = torch.randn(4, 3, 64, 64)
        with torch.no_grad():
            labels = model(images).argmax(1)
        labels[1::2] = 1 - labels[1::2]
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        paths = save_gradcam_examples(model, loader, self.output / "gradcam", num_samples=4)
        manifest = json.loads((self.output / "gradcam" / "manifest.json").read_text())
        self.assertEqual(len(paths), 4)
        self.assertEqual({row["category"] for row in manifest["examples"]}, {"correct", "wrong"})
        self.assertTrue(all(path.exists() for path in paths))

    def test_cli_train_and_gradcam_integration(self):
        config_path = self.output / "config.yaml"
        config_path.write_text(yaml.safe_dump({**self.config, "epochs": 1}), encoding="utf-8")
        loaders = self.loaders()
        with patch("src.data.build_dataloaders", return_value=loaders):
            with contextlib.redirect_stdout(io.StringIO()):
                history = main(["train", "--task", "classification", "--config", str(config_path)])
                paths = main(["gradcam", "--config", str(config_path), "--checkpoint", str(self.output / "checkpoints" / "best_scratch.pth"), "--num_samples", "2", "--max_scan_samples", "4"])
        self.assertEqual(len(history), 1)
        self.assertEqual(len(paths), 2)
        for filename in ("augmentations_scratch.png", "loss_scratch.png", "accuracy_scratch.png"):
            path = self.output / "figures" / filename
            with Image.open(path) as image:
                image.verify()

    def test_cli_help_and_extension_points(self):
        root = Path(__file__).resolve().parents[1]
        for args in (["--help"], ["train", "--help"], ["predict", "--help"], ["gradcam", "--help"]):
            result = subprocess.run([sys.executable, "-m", "src.cli", *args], cwd=root, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)
        register_model("test_encoder", ScratchCNN)
        self.assertIsInstance(build_model({**self.config, "model_type": "test_encoder"}), ScratchCNN)
        module = type(sys)("src.models.custom_cnn")
        module.CustomCNN = ScratchCNN
        with patch.dict(sys.modules, {"src.models.custom_cnn": module}):
            self.assertIsInstance(build_model({**self.config, "model_type": "custom"}), ScratchCNN)
        extension = type(sys)("src.training.tuning")
        extension.register_commands = lambda parsers: parsers.add_parser("tune")
        with patch.dict(sys.modules, {"src.training.tuning": extension}):
            self.assertEqual(build_parser().parse_args(["tune"]).command, "tune")
        with self.assertRaisesRegex(ValueError, "transfer_cnn"):
            build_model({**self.config, "model_type": "transfer"})

    def test_python_has_no_comments_or_docstrings(self):
        root = Path(__file__).resolve().parents[1]
        for folder in ("src", "tests"):
            for path in (root / folder).rglob("*.py"):
                with tokenize.open(path) as handle:
                    source = handle.read()
                tokens = tokenize.generate_tokens(io.StringIO(source).readline)
                self.assertFalse(any(token.type == tokenize.COMMENT for token in tokens), str(path))
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.assertIsNone(ast.get_docstring(node), str(path))


if __name__ == "__main__":
    unittest.main()
