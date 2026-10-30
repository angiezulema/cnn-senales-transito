"""Exporta el modelo final de HU07 para el sprint de despliegue (HU08).

Genera:
  final/saved_model/            -> TensorFlow SavedModel
  final/model.tflite            -> TFLite float32
  final/model_dynamic_range.tflite -> TFLite pesos cuantizados (I/O float)
  final/model_int8.tflite       -> (opcional --int8) cuantizacion completa
  final/class_names.json
  final/preprocessing.json
  final/export_report.json

Uso:
    python src/export_model.py --data-root C:/Users/angie/Downloads/HU03
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HU07_ROOT = Path(__file__).resolve().parent.parent
if str(HU07_ROOT) not in sys.path:
    sys.path.insert(0, str(HU07_ROOT))

import tensorflow as tf
from tensorflow import keras

from src.config import (
    CLASS_NAMES,
    DEFAULT_CONFIG,
    FINAL_DIR,
    PROJECT_ROOT,
    load_config,
    resolve_data_roots,
    resolve_path,
)
from src.data_loader import load_partition
from src.model import load_trained_model


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _representative_dataset(images: np.ndarray, n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=min(n, len(images)), replace=False)

    def generator():
        for i in idx:
            yield [images[i][None, ...].astype(np.float32) / 255.0]

    return generator


def _tflite_predict(tflite_path: Path, sample: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    x = sample.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    x = x[None, ...]

    if input_details["dtype"] == np.uint8:
        scale, zero = input_details["quantization"]
        x = (x / scale + zero).astype(np.uint8)
    else:
        x = x.astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details["index"])[0]

    if output_details["dtype"] == np.uint8:
        scale, zero = output_details["quantization"]
        out = (out.astype(np.float32) - zero) * scale
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exporta el modelo final HU07")
    parser.add_argument("--model", default=str(FINAL_DIR / "model_final.keras"))
    parser.add_argument("--config", default=str(FINAL_DIR / "config.json"))
    parser.add_argument("--out-dir", default=str(FINAL_DIR))
    parser.add_argument("--data-root", action="append", default=[])
    parser.add_argument("--int8", action="store_true", help="Genera modelo int8")
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=300,
        help="Muestras de test para verificar cada artefacto TFLite",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path) if cfg_path.exists() else load_config(DEFAULT_CONFIG)

    print(f"Cargando modelo: {model_path}")
    model = load_trained_model(str(model_path))

    report: dict = {
        "source_model": str(model_path),
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "num_params": int(model.count_params()),
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "artifacts": {},
    }

    # ── SavedModel ─────────────────────────────────────────────
    saved_model_dir = out_dir / "saved_model"
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    try:
        model.export(str(saved_model_dir))
        method = "model.export"
    except Exception:  # noqa: BLE001
        tf.saved_model.save(model, str(saved_model_dir))
        method = "tf.saved_model.save"
    report["artifacts"]["saved_model"] = {
        "path": str(saved_model_dir),
        "size_bytes": _dir_size(saved_model_dir),
        "method": method,
    }
    print(f"[ok] SavedModel ({method}) -> {saved_model_dir}")

    # ── TFLite float32 ─────────────────────────────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = out_dir / "model.tflite"
    tflite_path.write_bytes(tflite_model)
    report["artifacts"]["tflite_float32"] = {
        "path": str(tflite_path),
        "size_bytes": tflite_path.stat().st_size,
    }
    print(f"[ok] TFLite float32 -> {tflite_path}")

    # ── TFLite dynamic range (pesos cuantizados, I/O float) ────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    dyn_model = converter.convert()
    dyn_path = out_dir / "model_dynamic_range.tflite"
    dyn_path.write_bytes(dyn_model)
    report["artifacts"]["tflite_dynamic_range"] = {
        "path": str(dyn_path),
        "size_bytes": dyn_path.stat().st_size,
    }
    print(f"[ok] TFLite dynamic range -> {dyn_path}")

    # ── TFLite int8 (opcional) ─────────────────────────────────
    if args.int8:
        extra_roots = list(cfg["data"].get("extra_roots") or []) + list(args.data_root)
        roots = resolve_data_roots(extra_roots)
        train_csv = resolve_path(cfg["data"]["csv_train"], PROJECT_ROOT)
        X_train, _ = load_partition(
            train_csv,
            roots,
            height=int(cfg.get("img_height", 32)),
            width=int(cfg.get("img_width", 32)),
            num_classes=int(cfg.get("num_classes", 43)),
            use_cache=bool(cfg["data"].get("use_cache", True)),
            limit=300,
            verbose=False,
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset(
            X_train, seed=args.seed
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        int8_model = converter.convert()
        int8_path = out_dir / "model_int8.tflite"
        int8_path.write_bytes(int8_model)
        report["artifacts"]["tflite_int8"] = {
            "path": str(int8_path),
            "size_bytes": int8_path.stat().st_size,
        }
        print(f"[ok] TFLite int8 -> {int8_path}")

    # ── Contratos de despliegue ────────────────────────────────
    class_names = {str(i): name for i, name in CLASS_NAMES.items()}
    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)

    preprocessing = {
        "color_mode": "RGB",
        "img_height": int(cfg.get("img_height", 32)),
        "img_width": int(cfg.get("img_width", 32)),
        "resize_method": "bilinear",
        "scale": 1.0 / 255.0,
        "input_layout": "NHWC",
        "input_dtype": "float32",
        "output": "softmax probabilities (43 classes)",
    }
    with open(out_dir / "preprocessing.json", "w", encoding="utf-8") as f:
        json.dump(preprocessing, f, indent=2)

    # ── Verificacion sobre una muestra aleatoria del test ──────
    try:
        extra_roots = list(cfg["data"].get("extra_roots") or []) + list(args.data_root)
        roots = resolve_data_roots(extra_roots)
        test_csv = resolve_path(cfg["data"]["csv_test"], PROJECT_ROOT)
        X_test, y_test = load_partition(
            test_csv,
            roots,
            height=int(cfg.get("img_height", 32)),
            width=int(cfg.get("img_width", 32)),
            num_classes=int(cfg.get("num_classes", 43)),
            use_cache=bool(cfg["data"].get("use_cache", True)),
            verbose=False,
        )
        rng = np.random.default_rng(args.seed)
        n = min(int(args.verify_samples), len(X_test))
        idx = np.sort(rng.choice(len(X_test), size=n, replace=False))
        X_ver, y_ver = X_test[idx], y_test[idx]

        keras_prob = model.predict(
            X_ver.astype(np.float32) / 255.0, verbose=0
        )
        keras_pred = np.argmax(keras_prob, axis=1)
        verification: dict = {
            "samples": int(n),
            "seed": int(args.seed),
            "keras_accuracy": float((keras_pred == y_ver).mean()),
            "artifacts": {},
        }
        for name, artifact in report["artifacts"].items():
            if not name.startswith("tflite_"):
                continue
            path = Path(artifact["path"])
            prob = np.stack([_tflite_predict(path, x) for x in X_ver])
            pred = np.argmax(prob, axis=1)
            verification["artifacts"][name] = {
                "accuracy": float((pred == y_ver).mean()),
                "keras_agreement": float((pred == keras_pred).mean()),
                "mean_abs_prob_diff": float(np.abs(prob - keras_prob).mean()),
            }
        report["verification"] = verification
        print(f"[ok] Verificacion TFLite: {json.dumps(verification, indent=2)}")
    except Exception as exc:  # noqa: BLE001
        report["verification"] = {"error": str(exc)}
        print(f"[warn] No se pudo verificar TFLite: {exc}")

    with open(out_dir / "export_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nExportacion completa -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
