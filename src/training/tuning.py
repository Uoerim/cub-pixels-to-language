import csv
import gc
import json
import time
from pathlib import Path

import torch
import yaml
from sklearn.model_selection import ParameterGrid, ParameterSampler

from src.data import build_dataloaders
from src.data.dataset import load_records
from src.models import build_model
from src.training import Trainer, seed_everything


SEARCH_KEYS = {"lr", "weight_decay", "dropout", "batch_size"}


def trial_parameters(space, method, trials, seed):
    if not space or set(space) - SEARCH_KEYS:
        raise ValueError(f"Search a nonempty subset of {sorted(SEARCH_KEYS)}.")
    if any(not isinstance(values, list) or not values for values in space.values()):
        raise ValueError("Each search parameter needs a nonempty list of choices.")
    if trials < 1:
        raise ValueError("trials must be positive.")
    if method == "grid":
        return list(ParameterGrid(space))[:trials]
    if method == "random":
        count = min(trials, len(ParameterGrid(space)))
        return list(ParameterSampler(space, n_iter=count, random_state=seed))
    raise ValueError("Use grid or random search.")


def run_search(config, space, method="both", trials=4, train_data=None, test_data=None):
    methods = ["grid", "random"] if method == "both" else [method]
    plans = {name: trial_parameters(space, name, trials, config.get("seed", 42)) for name in methods}
    output = Path(config.get("save_dir", "outputs")) / "tuning"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Tuning output already exists; choose a fresh save_dir.")
    train_data, test_data = load_records(config, train_data, test_data)
    output.mkdir(parents=True, exist_ok=True)
    (output / "search_space.json").write_text(json.dumps({"space": space, "budget_per_method": trials, "seed": config.get("seed", 42)}, indent=2), encoding="utf-8")
    summaries, rows, best_config = [], [], None
    best_score = -float("inf")
    for name, parameters in plans.items():
        started = time.perf_counter()
        method_rows = []
        for index, values in enumerate(parameters):
            trial_config = {**config, **values, "save_dir": str(output / name / f"trial_{index:03d}"), "run_name": f"{config.get('model_type', 'scratch')}_{name}_{index:03d}", "search_method": name, "trial_index": index}
            if "dropout" in values and "dropout" in (config.get("model_kwargs") or {}):
                trial_config["model_kwargs"] = {**config["model_kwargs"], "dropout": values["dropout"]}
            seed_everything(config.get("seed", 42))
            loaders = build_dataloaders(trial_config, train_data, test_data)
            trainer = Trainer(build_model(trial_config), trial_config, loaders["train"].dataset.class_names)
            trial_started = time.perf_counter()
            trainer.fit(loaders["train"], loaders["val"])
            row = {"method": name, "trial": index, "val_top1": trainer.best_accuracy, "seconds": time.perf_counter() - trial_started, "run_id": trainer.tracker.run_id, "checkpoint": str(trainer.checkpoint_dir / f"best_{trainer.model_type}.pth"), "parameters": values}
            rows.append(row)
            method_rows.append(row)
            (output / "trials.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            if trainer.best_accuracy > best_score:
                best_score, best_config = trainer.best_accuracy, trial_config
            del trainer, loaders
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        best = max(method_rows, key=lambda row: row["val_top1"])
        summaries.append({"method": name, "trials": len(method_rows), "best_val_top1": best["val_top1"], "total_seconds": time.perf_counter() - started, "best_checkpoint": best["checkpoint"], "best_parameters": best["parameters"]})
    result = {"selection_metric": "validation top-1 (%)", "test_used_for_selection": False, "methods": summaries}
    (output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method", "trials", "best_val_top1", "total_seconds", "best_checkpoint"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    (output / "best_trial.yaml").write_text(yaml.safe_dump(best_config, sort_keys=False), encoding="utf-8")
    from src.visualization.comparison import save_search_comparison

    save_search_comparison(summaries, output / "comparison.png")
    return result


def tune_command(args):
    from src.cli import load_config

    config = load_config(args.config)
    with Path(args.search_space).open(encoding="utf-8") as handle:
        space = yaml.safe_load(handle)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    result = run_search(config, space, args.method, args.trials)
    print(json.dumps(result, indent=2))
    return result


def register_commands(subparsers):
    parser = subparsers.add_parser("tune", help="Compare grid and random search on validation only")
    parser.add_argument("--task", choices=["classification"], default="classification")
    parser.add_argument("--config", required=True)
    parser.add_argument("--search-space", required=True)
    parser.add_argument("--method", choices=["grid", "random", "both"], default="both")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--epochs", type=int)
    parser.set_defaults(handler=tune_command)
