"""Entrena un unico experimento de HU07 y guarda sus artefactos.

Uso:
    python src/train_experiment.py --exp-id E01 --epochs 12 \
        --override learning_rate=0.0005 --data-root C:/ruta/HU03
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HU07_ROOT = Path(__file__).resolve().parent.parent
if str(HU07_ROOT) not in sys.path:
    sys.path.insert(0, str(HU07_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow import keras

from src.config import (
    DEFAULT_CONFIG,
    EXPERIMENTS_DIR,
    PROJECT_ROOT,
    apply_overrides,
    config_hash,
    ensure_dirs,
    load_config,
    resolve_data_roots,
    resolve_path,
    set_seed,
)
from src.data_loader import (
    compute_class_weights,
    dataset_summary,
    load_partition,
    make_tf_dataset,
)
from src.model import build_cnn, compile_model


def plot_history(history: dict, save_path: Path, title: str = "") -> None:
    epochs = range(1, len(history.get("accuracy", [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["accuracy"], label="Entrenamiento")
    ax1.plot(epochs, history["val_accuracy"], label="Validacion")
    ax1.set_title(f"Accuracy por epoca {title}".strip())
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["loss"], label="Entrenamiento")
    ax2.plot(epochs, history["val_loss"], label="Validacion")
    ax2.set_title(f"Loss por epoca {title}".strip())
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def run_experiment(
    cfg: dict,
    exp_dir: str | Path,
    data_roots: list[Path],
    limit_train: int | None = None,
    limit_val: int | None = None,
    save_model: bool = True,
    model_filename: str = "model.keras",
    verbose: int = 2,
) -> dict:
    """Ejecuta un entrenamiento completo y devuelve sus metricas."""
    ensure_dirs()
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    exp_id = cfg.get("_exp_id") or exp_dir.name
    cfg = dict(cfg)
    cfg.pop("_exp_id", None)

    # Config serializada: incluye las raices resueltas para que evaluate/export
    # posteriores encuentren las imagenes sin repetir --data-root.
    serial_cfg = json.loads(json.dumps(_jsonable(cfg)))
    serial_cfg.setdefault("data", {})["extra_roots"] = [str(r) for r in data_roots]

    set_seed(
        int(cfg.get("seed", 42)),
        bool(cfg.get("deterministic", False)),
        intra_threads=cfg.get("intra_threads"),
        inter_threads=cfg.get("inter_threads"),
    )

    train_csv = resolve_path(cfg["data"]["csv_train"], PROJECT_ROOT)
    val_csv = resolve_path(cfg["data"]["csv_val"], PROJECT_ROOT)
    img_h = int(cfg.get("img_height", 32))
    img_w = int(cfg.get("img_width", 32))
    num_classes = int(cfg.get("num_classes", 43))
    use_cache = bool(cfg["data"].get("use_cache", True))

    print(f"\n=== {exp_id} ===")
    print(f"train: {train_csv}")
    print(f"val:   {val_csv}")
    print(f"roots: {[str(r) for r in data_roots]}")

    X_train, y_train = load_partition(
        train_csv,
        data_roots,
        height=img_h,
        width=img_w,
        num_classes=num_classes,
        use_cache=use_cache,
        limit=limit_train,
        seed=int(cfg.get("seed", 42)),
    )
    X_val, y_val = load_partition(
        val_csv,
        data_roots,
        height=img_h,
        width=img_w,
        num_classes=num_classes,
        use_cache=use_cache,
        limit=limit_val,
        seed=int(cfg.get("seed", 42)),
    )

    batch_size = int(cfg.get("batch_size", 64))
    train_ds = make_tf_dataset(
        X_train, y_train, batch_size=batch_size, shuffle=True,
        seed=int(cfg.get("seed", 42)),
    )
    val_ds = make_tf_dataset(
        X_val, y_val, batch_size=batch_size, shuffle=False,
    )

    class_weights = compute_class_weights(y_train, num_classes)

    model = build_cnn(cfg)
    compile_model(model, cfg)
    if verbose >= 1:
        model.summary()

    es_cfg = cfg.get("early_stopping", {})
    rl_cfg = cfg.get("reduce_lr", {})
    monitor = str(es_cfg.get("monitor", "val_accuracy"))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=int(es_cfg.get("patience", 7)),
            min_delta=float(es_cfg.get("min_delta", 1e-4)),
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=str(rl_cfg.get("monitor", "val_loss")),
            factor=float(rl_cfg.get("factor", 0.5)),
            patience=int(rl_cfg.get("patience", 3)),
            min_lr=float(rl_cfg.get("min_lr", 1e-6)),
            verbose=1,
        ),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(str(exp_dir / "log.csv"), append=False),
    ]

    best_model_path = exp_dir / model_filename
    if save_model:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(best_model_path),
                monitor=monitor,
                save_best_only=True,
                verbose=1,
            )
        )

    epochs = int(cfg.get("epochs", 50))
    print(f"Iniciando entrenamiento: {epochs} epocas max, batch={batch_size}")
    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=verbose,
    )
    elapsed = time.time() - start

    if save_model:
        model.save(str(best_model_path))

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    y_proba = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist["training_time_seconds"] = float(elapsed)

    def _best(metric_key: str, mode: str = "max"):
        vals = hist.get(metric_key)
        if not vals:
            return None, None
        idx = int(np.argmax(vals) if mode == "max" else np.argmin(vals))
        return idx + 1, float(vals[idx])

    best_epoch_acc, best_val_acc = _best("val_accuracy", "max")
    best_epoch_loss, best_val_loss = _best("val_loss", "min")

    metrics = {
        "status": "completed",
        "exp_id": exp_id,
        "config_hash": config_hash(cfg),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "epochs_ran": len(hist.get("loss", [])),
        "epochs_max": epochs,
        "best_epoch_val_accuracy": best_epoch_acc,
        "best_val_accuracy": best_val_acc,
        "best_epoch_val_loss": best_epoch_loss,
        "best_val_loss": best_val_loss,
        "final_val_accuracy": float(val_acc),
        "final_val_loss": float(val_loss),
        "val_f1_macro": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        "val_precision_macro": float(
            precision_score(y_val, y_pred, average="macro", zero_division=0)
        ),
        "val_recall_macro": float(
            recall_score(y_val, y_pred, average="macro", zero_division=0)
        ),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "train_summary": dataset_summary(y_train),
        "val_summary": dataset_summary(y_val),
        "training_time_seconds": float(elapsed),
        "num_params": int(model.count_params()),
        "data_roots": [str(r) for r in data_roots],
        "config": serial_cfg,
    }

    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(serial_cfg, f, indent=2, ensure_ascii=False)
    with open(exp_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(hist), f, indent=2)
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2, ensure_ascii=False)

    plot_history(hist, exp_dir / "curves.png", title=f"({exp_id})")

    minutes, seconds = divmod(int(elapsed), 60)
    print(
        f"[{exp_id}] completado en {minutes}m {seconds}s | "
        f"val_acc={val_acc:.4f} f1_macro={metrics['val_f1_macro']:.4f} "
        f"val_loss={val_loss:.4f}"
    )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena un experimento de HU07"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--override", action="append", default=[],
        help="Override tipo clave=valor (repetible)",
    )
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--exp-dir", default=None)
    parser.add_argument(
        "--data-root", action="append", default=[],
        help="Raiz adicional con archive/ o GTSRB/ (repetible)",
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.patience is not None:
        cfg.setdefault("early_stopping", {})["patience"] = args.patience

    exp_id = args.exp_id or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
    cfg["_exp_id"] = exp_id
    exp_dir = Path(args.exp_dir) if args.exp_dir else EXPERIMENTS_DIR / exp_id

    extra_roots = list(cfg["data"].get("extra_roots") or []) + list(args.data_root)
    roots = resolve_data_roots(extra_roots)

    run_experiment(
        cfg=cfg,
        exp_dir=exp_dir,
        data_roots=roots,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        save_model=not args.no_save_model,
        verbose=0 if args.quiet else 2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
