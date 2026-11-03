import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import ImageOps

from src.data.preprocess import caption_anomalies


def summarize(values):
    if not values:
        return {}
    array = np.asarray(values)
    return {"min": float(array.min()), "median": float(np.median(array)), "mean": float(array.mean()), "max": float(array.max())}


def analyze_records(records, split, output_dir, num_classes=200):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = np.zeros(num_classes, dtype=int)
    widths, heights, ratios, lengths, caption_counts = [], [], [], [], []
    modes, vocabulary, issues = Counter(), Counter(), Counter()
    sums, squares, pixels = np.zeros(3), np.zeros(3), 0
    seen = {"id": {}, "file_name": {}, "pixels": {}}
    duplicates, invalid, keep, rows, image_keys = [], [], [], [], []
    sample_images = []
    with (output_dir / f"captions_{split}.jsonl").open("w", encoding="utf-8") as cleaned:
        for index in range(len(records)):
            try:
                row = records[index]
                label = int(row["label"])
                if not 0 <= label < num_classes:
                    raise ValueError("Label outside configured range")
                image = ImageOps.exif_transpose(row["image"])
                image.load()
                modes[image.mode] += 1
                image = image.convert("RGB")
                if min(image.size) < 1:
                    raise ValueError("Empty image")
                array = np.asarray(image, dtype=np.float64) / 255
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                image_keys.append({"index": index, "hash": digest, "label": label})
                counts[label] += 1
                widths.append(image.width)
                heights.append(image.height)
                ratios.append(image.width / image.height)
                sums += array.sum(axis=(0, 1))
                squares += (array ** 2).sum(axis=(0, 1))
                pixels += image.width * image.height
                captions, flags = caption_anomalies(row.get("description"))
                caption_counts.append(len(captions))
                issues.update(flags)
                for caption in captions:
                    tokens = caption.split()
                    lengths.append(len(tokens))
                    vocabulary.update(tokens)
                cleaned.write(json.dumps({"index": index, "id": row.get("id"), "file_name": row.get("file_name"), "captions": captions, "issues": flags}, ensure_ascii=False) + "\n")
                duplicate_pixel = False
                for key, value in (("id", row.get("id")), ("file_name", row.get("file_name")), ("pixels", digest)):
                    if value is None or value == "":
                        continue
                    value = str(value)
                    if value in seen[key]:
                        previous, previous_label = seen[key][value]
                        duplicates.append({"kind": key, "index": index, "original": previous, "conflicting_label": previous_label != label})
                        duplicate_pixel |= key == "pixels"
                    else:
                        seen[key][value] = (index, label)
                if not duplicate_pixel:
                    keep.append(index)
                if len(sample_images) < 8:
                    sample_images.append((image.copy(), label, captions[0] if captions else "Missing caption"))
                rows.append({"index": index, "label": label, "width": image.width, "height": image.height, "aspect_ratio": ratios[-1], "captions": len(captions)})
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                invalid.append({"index": index, "error": f"{type(error).__name__}: {error}"})
    mean = sums / max(pixels, 1)
    std = np.sqrt(np.maximum(0, squares / max(pixels, 1) - mean ** 2))
    positive = counts[counts > 0]
    report = {
        "split": split, "records": len(records), "valid_images": len(rows), "class_counts": counts.tolist(),
        "missing_classes": np.flatnonzero(counts == 0).tolist(),
        "imbalance_ratio": float(positive.max() / positive.min()) if len(positive) else None,
        "width": summarize(widths), "height": summarize(heights), "aspect_ratio": summarize(ratios),
        "original_modes": dict(modes), "rgb_mean": mean.tolist(), "rgb_std": std.tolist(), "pixel_count": pixels,
        "caption_word_length": summarize(lengths), "captions_per_image": summarize(caption_counts),
        "vocabulary_size": len(vocabulary), "caption_issues": dict(issues),
        "rare_token_candidates": sorted(word for word, count in vocabulary.items() if count == 1),
        "top_tokens": vocabulary.most_common(40), "duplicates": duplicates, "invalid_records": invalid,
        "keep_indices": keep, "image_hashes": image_keys,
        "policy": "Exclude unreadable/invalid training records and repeated decoded training pixels; review conflicting labels. Never remove official test records. Rare tokens are review candidates, not automatic spelling corrections.",
    }
    (output_dir / f"eda_{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / f"images_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "label", "width", "height", "aspect_ratio", "captions"])
        writer.writeheader()
        writer.writerows(rows)
    from src.visualization.eda import save_eda_plots

    save_eda_plots(report, widths, heights, ratios, lengths, caption_counts, sample_images, output_dir)
    return report


def run_eda(config, train_data=None, test_data=None):
    from src.data.dataset import load_records

    train_data, test_data = load_records(config, train_data, test_data)
    output = Path(config.get("save_dir", "outputs")) / "eda"
    train = analyze_records(train_data, "train", output, config.get("num_classes", 200))
    test = analyze_records(test_data, "test", output, config.get("num_classes", 200))
    test_hashes = {row["hash"] for row in test["image_hashes"]}
    overlaps = [row["index"] for row in train["image_hashes"] if row["hash"] in test_hashes]
    fingerprint = getattr(train_data, "_fingerprint", None)
    manifest = {
        "train_fingerprint": fingerprint, "train_records": len(train_data), "test_records": len(test_data),
        "test_fingerprint": getattr(test_data, "_fingerprint", None),
        "train_keep_indices": [index for index in train["keep_indices"] if index not in set(overlaps)],
        "train_test_overlap_indices": overlaps, "test_invalid_records": test["invalid_records"],
        "conflicting_labels": [row for row in train["duplicates"] if row["kind"] == "pixels" and row["conflicting_label"]],
    }
    (output / "cleaning_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def eda_command(args):
    from src.cli import load_config

    result = run_eda(load_config(args.config))
    print(json.dumps(result, indent=2))
    return result


def register_commands(subparsers):
    parser = subparsers.add_parser("eda", help="Audit images and captions and export cleaning decisions")
    parser.add_argument("--config", required=True)
    parser.set_defaults(handler=eda_command)
