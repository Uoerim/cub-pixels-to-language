import random
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from PIL import ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.preprocess import preprocess_captions


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CUBDataset(Dataset):
    def __init__(self, records, transform=None, indices=None):
        self.records = records
        self.transform = transform
        self.indices = list(range(len(records))) if indices is None else list(indices)
        feature = getattr(records, "features", {}).get("label")
        self.class_names = list(getattr(feature, "names", []) or [])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.records[self.indices[index]]
        image = ImageOps.exif_transpose(row["image"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])

    @staticmethod
    def preprocess_captions(description):
        return preprocess_captions(description)

    def captions(self, index):
        row = self.records[self.indices[index]]
        return self.preprocess_captions(row.get("description", ""))


def build_transforms(img_size=224):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize,
    ])
    test = transforms.Compose([
        transforms.Resize(round(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train, test


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def load_records(config, train_data=None, test_data=None):
    cache_dir = config.get("cache_dir")
    local = config.get("dataset_dir")
    if train_data is None:
        train_data = load_from_disk(str(Path(local) / "train")) if local else load_dataset("Multimodal-Fatima/CUB_train", split="train", cache_dir=cache_dir)
    if test_data is None:
        test_data = load_from_disk(str(Path(local) / "test")) if local else load_dataset("Multimodal-Fatima/CUB_test", split="test", cache_dir=cache_dir)
    return train_data, test_data


def build_dataloaders(config, train_data=None, test_data=None):
    train_data, test_data = load_records(config, train_data, test_data)
    train_names = getattr(getattr(train_data, "features", {}).get("label"), "names", None)
    test_names = getattr(getattr(test_data, "features", {}).get("label"), "names", None)
    if train_names and test_names and train_names != test_names:
        raise ValueError("Train and test class mappings differ.")
    labels = np.asarray(train_data["label"])
    num_classes = config.get("num_classes", 200)
    for values in (labels, np.asarray(test_data["label"])):
        if len(values) == 0 or values.min() < 0 or values.max() >= num_classes:
            raise ValueError("Dataset labels must be within the configured class range.")
    val_split = config.get("val_split", 0.15)
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1.")
    indices = np.arange(len(labels))
    if config.get("data_manifest"):
        manifest = json.loads(Path(config["data_manifest"]).read_text(encoding="utf-8"))
        required = {"train_records", "test_records", "train_fingerprint", "test_fingerprint", "test_invalid_records", "conflicting_labels", "train_keep_indices"}
        if not isinstance(manifest, dict) or not required.issubset(manifest):
            raise ValueError("Invalid cleaning manifest; rerun EDA.")
        if manifest["train_records"] != len(train_data) or manifest["test_records"] != len(test_data) or manifest["train_fingerprint"] != getattr(train_data, "_fingerprint", None) or manifest["test_fingerprint"] != getattr(test_data, "_fingerprint", None):
            raise ValueError("Cleaning manifest does not match this dataset; rerun EDA.")
        if manifest["test_invalid_records"] or manifest["conflicting_labels"]:
            raise ValueError("Review invalid test records or conflicting training labels before training.")
        indices = np.asarray(manifest["train_keep_indices"], dtype=int)
        if len(indices) == 0 or len(set(indices.tolist())) != len(indices) or indices.min() < 0 or indices.max() >= len(labels):
            raise ValueError("Invalid training indices in cleaning manifest.")
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        random_state=config.get("seed", 42),
        stratify=labels[indices],
    )
    train_transform, test_transform = build_transforms(config.get("img_size", 224))
    datasets = {
        "train": CUBDataset(train_data, train_transform, train_indices),
        "val": CUBDataset(train_data, test_transform, val_indices),
        "test": CUBDataset(test_data, test_transform),
    }
    workers = config.get("num_workers", 0)
    loaders = {}
    for name, dataset in datasets.items():
        generator = torch.Generator().manual_seed(config.get("seed", 42))
        loaders[name] = DataLoader(
            dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=name == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available() and config.get("device", "auto") != "cpu",
            persistent_workers=workers > 0,
            worker_init_fn=seed_worker,
            generator=generator,
        )
    return loaders
