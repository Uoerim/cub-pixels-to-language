import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.eda import analyze_records, run_eda
from src.data.preprocess import caption_anomalies, preprocess_captions


class DataAuditTests(unittest.TestCase):
    def test_text_normalization_and_malformed_input(self):
        self.assertEqual(preprocess_captions("  ＢLUE-gray_bird &amp; wing.\r\n\nA\tbeak!"), ["blue gray bird wing", "a beak"])
        self.assertEqual(preprocess_captions(None), [])
        self.assertEqual(preprocess_captions(12), [])
        self.assertIn("non_string_description", caption_anomalies(12)[1])
        self.assertIn("repeated_caption", caption_anomalies("bird\nbird")[1])

    def test_audit_detects_duplicates_corruption_and_weighted_channels(self):
        records = [
            {"id": 1, "file_name": "a", "label": 0, "image": Image.new("RGB", (4, 2), (255, 0, 0)), "description": "red bird\nred bird"},
            {"id": 1, "file_name": "a", "label": 0, "image": Image.new("RGB", (4, 2), (255, 0, 0)), "description": None},
            {"id": 3, "file_name": "b", "label": 1, "image": Image.new("RGB", (2, 2), (0, 0, 255)), "description": "Blue bird 123"},
            {"label": 1, "image": None},
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = analyze_records(records, "train", directory, 2)
            self.assertEqual(report["class_counts"], [2, 1])
            self.assertEqual(report["keep_indices"], [0, 2])
            self.assertEqual(len(report["invalid_records"]), 1)
            self.assertEqual({row["kind"] for row in report["duplicates"]}, {"id", "file_name", "pixels"})
            self.assertAlmostEqual(report["rgb_mean"][0], 0.8)
            self.assertAlmostEqual(report["rgb_mean"][2], 0.2)
            self.assertEqual(report["vocabulary_size"], 4)
            self.assertTrue((Path(directory) / "statistics_train.png").exists())
            self.assertEqual(len((Path(directory) / "captions_train.jsonl").read_text().splitlines()), 3)

    def test_cross_split_overlap_is_removed_only_from_training(self):
        row = {"label": 0, "image": Image.new("RGB", (4, 4), "red"), "description": "red bird"}
        with tempfile.TemporaryDirectory() as directory:
            result = run_eda({"save_dir": directory, "num_classes": 1}, [row], [row])
            self.assertEqual(result["train_keep_indices"], [])
            self.assertEqual(result["test_records"], 1)
            self.assertEqual(json.loads((Path(directory) / "eda" / "cleaning_manifest.json").read_text()), result)

    def test_cleaning_manifest_prevents_stale_data_and_split_leakage(self):
        from src.data import build_dataloaders
        from tests.test_pipeline import make_records

        train, test = make_records(), make_records(4)
        with tempfile.TemporaryDirectory() as directory:
            config = {"save_dir": directory, "num_classes": 2, "val_split": 0.25, "img_size": 64}
            manifest = run_eda(config, train, test)
            config["data_manifest"] = str(Path(directory) / "eda" / "cleaning_manifest.json")
            loaders = build_dataloaders(config, train, test)
            selected = set(loaders["train"].dataset.indices) | set(loaders["val"].dataset.indices)
            self.assertEqual(selected, set(manifest["train_keep_indices"]))
            self.assertFalse(selected & set(manifest["train_test_overlap_indices"]))
            self.assertEqual(len(loaders["test"].dataset), 4)
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_dataloaders(config, train, test.select([3, 2, 1, 0]))
