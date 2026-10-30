"""Evaluacion del modelo final de HU07 sobre el conjunto oficial de prueba.

Uso:
    python src/evaluate_final.py --data-root C:/Users/angie/Downloads/HU03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HU07_ROOT = Path(__file__).resolve().parent.parent
if str(HU07_ROOT) not in sys.path:
    sys.path.insert(0, str(HU07_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow import keras

from src.config import (
    CLASS_NAMES,
    DEFAULT_CONFIG,
    FINAL_DIR,
    PROJECT_ROOT,
    ensure_dirs,
    load_config,
    resolve_data_roots,
    resolve_path,
)
from src.data_loader import load_partition, make_tf_dataset
from src.model import load_trained_model


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, names: dict) -> None:
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = [names.get(i, str(i)) for i in range(len(cm))]
    ax.set(
        xticks=np.arange(len(cm)),
        yticks=np.arange(len(cm)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Etiqueta real",
        xlabel="Prediccion",
        title="Matriz de confusion - Conjunto de prueba",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluacion final HU07")
    parser.add_argument(
        "--model", default=str(FINAL_DIR / "model_final.keras")
    )
    parser.add_argument(
        "--config", default=str(FINAL_DIR / "config.json")
    )
    parser.add_argument("--data-root", action="append", default=[])
    parser.add_argument("--out-dir", default=str(FINAL_DIR))
    parser.add_argument("--limit-test", type=int, default=None)
    args = parser.parse_args(argv)

    ensure_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else load_config(DEFAULT_CONFIG)

    extra_roots = list(cfg["data"].get("extra_roots") or []) + list(args.data_root)
    roots = resolve_data_roots(extra_roots)

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}. Ejecuta final_train.py primero."
        )

    print(f"Cargando modelo: {model_path}")
    model = load_trained_model(str(model_path))

    test_csv = resolve_path(cfg["data"]["csv_test"], PROJECT_ROOT)
    X_test, y_test = load_partition(
        test_csv,
        roots,
        height=int(cfg.get("img_height", 32)),
        width=int(cfg.get("img_width", 32)),
        num_classes=int(cfg.get("num_classes", 43)),
        use_cache=bool(cfg["data"].get("use_cache", True)),
        limit=args.limit_test,
    )
    batch_size = int(cfg.get("batch_size", 64))
    test_ds = make_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

    print("Generando predicciones...")
    y_proba = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    metrics = {
        "model": str(model_path),
        "selected_from": cfg.get("_meta", {}).get("selected_from"),
        "total_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
    }

    print("\n=== Resultados sobre test ===")
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"):
        print(f"  {key:18s}: {metrics[key]:.4f}")
    print(f"  total_samples     : {metrics['total_samples']}")

    target_names = [CLASS_NAMES.get(i, f"Clase {i}") for i in range(43)]
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(43),
        target_names=target_names,
        zero_division=0,
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(43))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png", CLASS_NAMES)

    print(f"\nMetricas guardadas en {out_dir / 'test_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
