import random
import re

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
        image = row["image"].convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])

    @staticmethod
    def preprocess_captions(description):
        captions = []
        for line in (description or "").split("\n"):
            caption = re.sub(r"[\W_]+", " ", line.lower()).strip()
            if caption:
                captions.append(caption)
        return captions

    def captions(self, index):
        row = self.records[self.indices[index]]
        return self.preprocess_captions(row.get("description", ""))


def build_transforms(img_size=224):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
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


def build_dataloaders(config, train_data=None, test_data=None):
    cache_dir = config.get("cache_dir")
    if train_data is None:
        train_data = load_dataset("Multimodal-Fatima/CUB_train", split="train", cache_dir=cache_dir)
    if test_data is None:
        test_data = load_dataset("Multimodal-Fatima/CUB_test", split="test", cache_dir=cache_dir)
    labels = np.asarray(train_data["label"])
    num_classes = config.get("num_classes", 200)
    for values in (labels, np.asarray(test_data["label"])):
        if len(values) == 0 or values.min() < 0 or values.max() >= num_classes:
            raise ValueError("Dataset labels must be within the configured class range.")
    val_split = config.get("val_split", 0.15)
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1.")
    train_indices, val_indices = train_test_split(
        np.arange(len(labels)),
        test_size=val_split,
        random_state=config.get("seed", 42),
        stratify=labels,
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
