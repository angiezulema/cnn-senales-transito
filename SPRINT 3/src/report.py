"""Genera tabla comparativa y graficos a partir de los experimentos de HU07.

Uso:
    python src/report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

HU07_ROOT = Path(__file__).resolve().parent.parent
if str(HU07_ROOT) not in sys.path:
    sys.path.insert(0, str(HU07_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import DOCS_DIR, EXPERIMENTS_DIR, FINAL_DIR, RESULTS_DIR, ensure_dirs

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

FIELDNAMES = [
    "exp_id", "status", "val_accuracy", "val_f1_macro", "val_loss",
    "best_epoch", "epochs_ran", "training_time_seconds", "num_params",
    "config_hash", *SPACE_KEYS,
]

HU06_REFERENCE = {
    "name": "HU06 (referencia)",
    "val_accuracy": 0.9995,
    "val_loss": 0.0028,
    "epochs": 34,
}


def load_experiments(experiments_dir: Path) -> list[dict]:
    metrics: list[dict] = []
    if not experiments_dir.exists():
        return metrics
    for exp_dir in sorted(experiments_dir.iterdir()):
        path = exp_dir / "metrics.json"
        if exp_dir.is_dir() and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("status") == "completed":
                    metrics.append(data)
            except (json.JSONDecodeError, OSError):
                continue
    return metrics


def to_row(metrics: dict) -> dict:
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


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            r.get("val_accuracy") is None,
            -(r.get("val_accuracy") or 0.0),
            -(r.get("val_f1_macro") or 0.0),
            r.get("training_time_seconds") or 0.0,
        ),
    )


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_comparison(rows: list[dict], path: Path) -> None:
    ordered = list(reversed(rows))
    labels = [r["exp_id"] for r in ordered]
    accs = [(r.get("val_accuracy") or 0.0) * 100 for r in ordered]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(5, 0.55 * len(rows) + 2)))

    ax1.barh(labels, accs, color="#3b82f6")
    ax1.axvline(HU06_REFERENCE["val_accuracy"] * 100, color="#ef4444",
                linestyle="--", linewidth=1.5, label="HU06 (99.95%)")
    ax1.set_xlabel("Val accuracy (%)")
    ax1.set_title("Comparacion de experimentos")
    ax1.set_xlim(left=max(0, min(accs) - 1), right=100.4)
    for i, value in enumerate(accs):
        ax1.text(value + 0.03, i, f"{value:.2f}", va="center", fontsize=7)
    ax1.legend(loc="lower right")
    ax1.grid(True, axis="x", alpha=0.3)

    times = [(r.get("training_time_seconds") or 0.0) / 60 for r in rows]
    vals = [(r.get("val_accuracy") or 0.0) * 100 for r in rows]
    ax2.scatter(times, vals, color="#10b981")
    for row, x, y in zip(rows, times, vals):
        ax2.annotate(row["exp_id"], (x, y), fontsize=6, xytext=(3, 3),
                     textcoords="offset points")
    ax2.set_xlabel("Tiempo de entrenamiento (min)")
    ax2.set_ylabel("Val accuracy (%)")
    ax2.set_title("Calidad vs costo computacional")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "# HU07 - Resultados de la busqueda de hiperparametros",
        "",
        f"> Generado: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Referencia HU06: val_accuracy **{HU06_REFERENCE['val_accuracy']:.4f}** "
        f"({HU06_REFERENCE['epochs']} epocas).",
        "",
        "| # | Experimento | Val acc | F1 macro | Val loss | Epoca | Epocas | Tiempo (min) | LR | Batch | Drop conv/dense | Dense | Optim | WD | LS |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(rows, start=1):
        lines.append(
            "| {i} | {exp} | {acc:.4f} | {f1:.4f} | {loss:.4f} | {be} | {ep} | "
            "{t:.1f} | {lr} | {bs} | {dc}/{dd} | {du} | {opt} | {wd} | {ls} |".format(
                i=i,
                exp=row["exp_id"],
                acc=row.get("val_accuracy") or 0.0,
                f1=row.get("val_f1_macro") or 0.0,
                loss=row.get("val_loss") or 0.0,
                be=row.get("best_epoch"),
                ep=row.get("epochs_ran"),
                t=(row.get("training_time_seconds") or 0.0) / 60,
                lr=row.get("learning_rate"),
                bs=row.get("batch_size"),
                dc=row.get("dropout_conv"),
                dd=row.get("dropout_dense"),
                du=row.get("dense_units"),
                opt=row.get("optimizer"),
                wd=row.get("weight_decay"),
                ls=row.get("label_smoothing"),
            )
        )

    if rows:
        best = rows[0]
        lines += [
            "",
            "## Mejor configuracion",
            "",
            f"- Experimento: **{best['exp_id']}**",
            f"- Val accuracy: **{best.get('val_accuracy'):.4f}**",
            f"- F1 macro: **{best.get('val_f1_macro'):.4f}**",
            f"- Val loss: {best.get('val_loss'):.4f}",
            f"- Epoca del mejor modelo: {best.get('best_epoch')} "
            f"de {best.get('epochs_ran')}",
        ]

    test_metrics_path = FINAL_DIR / "test_metrics.json"
    if test_metrics_path.exists():
        with open(test_metrics_path, "r", encoding="utf-8") as f:
            test = json.load(f)
        lines += [
            "",
            "## Evaluacion sobre el conjunto de prueba",
            "",
            "| Metrica | Valor |",
            "|---|---|",
            f"| Accuracy | {test.get('accuracy', 0):.4f} |",
            f"| Precision macro | {test.get('precision_macro', 0):.4f} |",
            f"| Recall macro | {test.get('recall_macro', 0):.4f} |",
            f"| F1 macro | {test.get('f1_macro', 0):.4f} |",
            f"| F1 weighted | {test.get('f1_weighted', 0):.4f} |",
            f"| Muestras | {test.get('total_samples')} |",
            "",
            "> El conjunto de prueba no se uso durante la busqueda ni el "
            "entrenamiento; solo en esta evaluacion final.",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reporte comparativo HU07")
    parser.add_argument("--experiments-dir", default=str(EXPERIMENTS_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--docs-dir", default=str(DOCS_DIR))
    args = parser.parse_args(argv)

    ensure_dirs()
    experiments_dir = Path(args.experiments_dir)
    results_dir = Path(args.results_dir)
    docs_dir = Path(args.docs_dir)

    metrics = load_experiments(experiments_dir)
    rows = sort_rows([to_row(m) for m in metrics])
    write_csv(rows, results_dir / "search_results.csv")

    if rows:
        plot_comparison(rows, results_dir / "comparison_val_accuracy.png")
    write_markdown(rows, docs_dir / "HU07_resultados.md")

    print(f"Experimentos encontrados: {len(rows)}")
    if rows:
        best = rows[0]
        print(
            f"Mejor: {best['exp_id']} val_acc={best['val_accuracy']:.4f} "
            f"f1={best['val_f1_macro']:.4f}"
        )
        print(f"Tabla: {results_dir / 'search_results.csv'}")
        print(f"Grafico: {results_dir / 'comparison_val_accuracy.png'}")
    print(f"Markdown: {docs_dir / 'HU07_resultados.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
