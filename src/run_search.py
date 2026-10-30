"""Random search de hiperparametros para HU07 (reanudable).

Uso tipico:
    python src/run_search.py --trials 8 --epochs 12 \
        --data-root C:/Users/angie/Downloads/HU03
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HU07_ROOT = Path(__file__).resolve().parent.parent
if str(HU07_ROOT) not in sys.path:
    sys.path.insert(0, str(HU07_ROOT))

from src.config import (
    DEFAULT_CONFIG,
    DEFAULT_SPACE,
    EXPERIMENTS_DIR,
    RESULTS_DIR,
    apply_overrides,
    config_hash,
    ensure_dirs,
    load_config,
    load_yaml,
    resolve_data_roots,
    set_seed,
)
from src.train_experiment import run_experiment

SPACE_KEYS = [
    "learning_rate",
    "batch_size",
    "dropout_conv",
    "dropout_dense",
    "dense_units",
    "optimizer",
    "weight_decay",
    "label_smoothing",
]


def _log(message: str, log_path: Path | None = None) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def sample_config(rng: random.Random, base: dict, space: dict) -> dict:
    cfg = copy.deepcopy(base)
    for key, values in space.items():
        if not values:
            continue
        cfg[key] = rng.choice(list(values))
    if str(cfg.get("optimizer", "adam")).lower() == "adam":
        cfg["weight_decay"] = 0.0
    return cfg


def _signature(cfg: dict) -> tuple:
    values = []
    for key in SPACE_KEYS:
        value = cfg.get(key)
        if isinstance(value, list):
            value = tuple(value)
        values.append(value)
    return tuple(values)


def generate_trials(
    base: dict, space: dict, n_trials: int, seed: int, include_baseline: bool
) -> list[tuple[str, dict]]:
    rng = random.Random(seed)
    trials: list[tuple[str, dict]] = []
    seen: set[tuple] = set()

    if include_baseline:
        baseline = copy.deepcopy(base)
        trials.append(("E00_baseline", baseline))
        seen.add(_signature(baseline))

    max_attempts = n_trials * 100
    attempts = 0
    while len(trials) < n_trials and attempts < max_attempts:
        attempts += 1
        cfg = sample_config(rng, base, space)
        sig = _signature(cfg)
        if sig in seen:
            continue
        seen.add(sig)
        idx = len(trials)
        trials.append((f"E{idx:02d}_rnd", cfg))

    return trials


def _load_result(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _row_from_metrics(metrics: dict) -> dict:
    cfg = metrics.get("config", {})
    row = {
        "exp_id": metrics.get("exp_id"),
        "status": metrics.get("status"),
        "val_accuracy": metrics.get("best_val_accuracy"),
        "val_f1_macro": metrics.get("val_f1_macro"),
        "val_loss": metrics.get("best_val_loss"),
        "best_epoch": metrics.get("best_epoch_val_accuracy"),
        "epochs_ran": metrics.get("epochs_ran"),
        "training_time_seconds": metrics.get("training_time_seconds"),
        "num_params": metrics.get("num_params"),
        "config_hash": metrics.get("config_hash"),
    }
    for key in SPACE_KEYS:
        row[key] = cfg.get(key)
    return row


def write_results(results: list[dict], results_dir: Path) -> list[dict]:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "search_results.csv"
    fieldnames = [
        "exp_id", "status", "val_accuracy", "val_f1_macro", "val_loss",
        "best_epoch", "epochs_ran", "training_time_seconds", "num_params",
        "config_hash", "learning_rate", "batch_size", "dropout_conv",
        "dropout_dense", "dense_units", "optimizer", "weight_decay",
        "label_smoothing",
    ]
    rows = [_row_from_metrics(m) for m in results]
    rows.sort(
        key=lambda r: (
            r.get("val_accuracy") is None,
            -(r.get("val_accuracy") or 0.0),
            -(r.get("val_f1_macro") or 0.0),
            r.get("training_time_seconds") or 0.0,
        )
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Random search HU07")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--space", default=str(DEFAULT_SPACE))
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", action="append", default=[])
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--experiments-dir", default=str(EXPERIMENTS_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()
    experiments_dir = Path(args.experiments_dir)
    results_dir = Path(args.results_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "search.log"

    base = apply_overrides(load_config(args.config), args.override)
    space = load_yaml(args.space)

    extra_roots = list(base["data"].get("extra_roots") or []) + list(args.data_root)
    roots = resolve_data_roots(extra_roots)

    trials = generate_trials(
        base, space, args.trials, args.seed, not args.no_baseline
    )
    _log(
        f"Random search: {len(trials)} trials, epochs={args.epochs}, "
        f"patience={args.patience}, seed={args.seed}",
        log_path,
    )
    _log(f"Raices de datos: {[str(r) for r in roots]}", log_path)

    completed: list[dict] = []
    failures = 0
    started = time.time()

    for idx, (exp_id, cfg) in enumerate(trials, start=1):
        exp_dir = experiments_dir / exp_id
        metrics_path = exp_dir / "metrics.json"

        cfg = copy.deepcopy(cfg)
        cfg["epochs"] = args.epochs
        cfg.setdefault("early_stopping", {})["patience"] = args.patience

        existing = None if args.no_resume else _load_result(metrics_path)
        expected_hash = config_hash(cfg)
        if existing and existing.get("status") == "completed":
            if existing.get("config_hash") == expected_hash:
                _log(
                    f"({idx}/{len(trials)}) {exp_id}: ya completado, se omite",
                    log_path,
                )
                completed.append(existing)
                continue
            _log(
                f"({idx}/{len(trials)}) {exp_id}: config cambio, se reentrena",
                log_path,
            )

        cfg["_exp_id"] = exp_id

        _log(f"({idx}/{len(trials)}) {exp_id}: iniciando", log_path)
        try:
            metrics = run_experiment(
                cfg=cfg,
                exp_dir=exp_dir,
                data_roots=roots,
                limit_train=args.limit_train,
                limit_val=args.limit_val,
                save_model=args.save_models,
                verbose=0 if args.quiet else 2,
            )
            completed.append(metrics)
            _log(
                f"({idx}/{len(trials)}) {exp_id}: "
                f"val_acc={metrics['best_val_accuracy']:.4f} "
                f"f1={metrics['val_f1_macro']:.4f} "
                f"t={metrics['training_time_seconds'] / 60:.1f}m",
                log_path,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            _log(f"({idx}/{len(trials)}) {exp_id}: ERROR {exc}", log_path)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(traceback.format_exc() + "\n")

        if completed:
            write_results(completed, results_dir)

    if not completed:
        _log("No se completo ningun experimento. Revisa search.log.", log_path)
        return 1

    rows = write_results(completed, results_dir)
    best_row = rows[0]
    best_metrics = next(
        m for m in completed if m.get("exp_id") == best_row["exp_id"]
    )

    best_config = copy.deepcopy(best_metrics["config"])
    best_config["_meta"] = {
        "selected_from": best_row["exp_id"],
        "criterion": "best val_accuracy, tie-break val_f1_macro then training_time",
        "val_accuracy": best_row["val_accuracy"],
        "val_f1_macro": best_row["val_f1_macro"],
        "val_loss": best_row["val_loss"],
        "search_trials": len(trials),
        "search_epochs": args.epochs,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(results_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - started
    _log(
        f"Busqueda finalizada en {elapsed / 60:.1f} min. "
        f"Mejor: {best_row['exp_id']} val_acc={best_row['val_accuracy']:.4f}. "
        f"Config: {results_dir / 'best_config.json'}",
        log_path,
    )
    if failures:
        _log(f"Experimentos fallidos: {failures}", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
