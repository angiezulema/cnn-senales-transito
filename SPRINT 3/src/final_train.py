"""Entrenamiento final de HU07 con la mejor configuracion encontrada.

Uso:
    python src/final_train.py --config results/best_config.json --epochs 80 \
        --data-root C:/Users/angie/Downloads/HU03
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

HU07_ROOT = Path(__file__).resolve().parent.parent
if str(HU07_ROOT) not in sys.path:
    sys.path.insert(0, str(HU07_ROOT))

from src.config import (
    DEFAULT_CONFIG,
    FINAL_DIR,
    RESULTS_DIR,
    apply_overrides,
    ensure_dirs,
    load_config,
    resolve_data_roots,
)
from src.train_experiment import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrenamiento final HU07")
    default_cfg = RESULTS_DIR / "best_config.json"
    parser.add_argument(
        "--config",
        default=str(default_cfg if default_cfg.exists() else DEFAULT_CONFIG),
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--data-root", action="append", default=[])
    parser.add_argument("--final-dir", default=str(FINAL_DIR))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    ensure_dirs()
    cfg = load_config(args.config)
    cfg.pop("_meta", None)
    cfg = apply_overrides(cfg, args.override)
    cfg["epochs"] = args.epochs
    cfg.setdefault("early_stopping", {})["patience"] = args.patience
    cfg["_exp_id"] = "final"

    extra_roots = list(cfg["data"].get("extra_roots") or []) + list(args.data_root)
    roots = resolve_data_roots(extra_roots)

    final_dir = Path(args.final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_experiment(
        cfg=cfg,
        exp_dir=final_dir,
        data_roots=roots,
        save_model=True,
        model_filename="model_final.keras",
        verbose=0 if args.quiet else 2,
    )

    # Copia de trazabilidad de la configuracion usada
    cfg_serial = json.loads(json.dumps(cfg, default=str))
    cfg_serial.setdefault("data", {})["extra_roots"] = [str(r) for r in roots]
    with open(final_dir / "best_config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg_serial, f, indent=2, ensure_ascii=False)

    if Path(args.config).exists() and Path(args.config) != final_dir / "config.json":
        try:
            shutil.copyfile(args.config, final_dir / "source_config.json")
        except OSError:
            pass

    print(
        f"\nModelo final: {final_dir / 'model_final.keras'}\n"
        f"val_accuracy={metrics['best_val_accuracy']:.4f} "
        f"f1_macro={metrics['val_f1_macro']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
