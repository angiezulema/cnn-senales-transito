"""Verifica el entorno y los datos antes de entrenar HU07.

Uso:
    python setup/check_env.py --data-root C:/Users/angie/Downloads/HU03
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import shutil
import sys
from pathlib import Path

HU07_ROOT = Path(__file__).resolve().parent.parent
if str(HU07_ROOT) not in sys.path:
    sys.path.insert(0, str(HU07_ROOT))

from src.config import (
    CLASS_NAMES,
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    load_config,
    resolve_data_roots,
    resolve_path,
)

OK = "[OK]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def check(name: str, condition: bool, detail: str, critical: bool = False) -> bool:
    status = OK if condition else (FAIL if critical else WARN)
    print(f"{status:6s} {name}: {detail}")
    return condition


def total_ram_gb() -> float | None:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / (1024**3)
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check de entorno HU07")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root", action="append", default=[])
    parser.add_argument("--sample", type=int, default=100)
    args = parser.parse_args(argv)

    print("=" * 68)
    print("HU07 - Verificacion de entorno")
    print("=" * 68)
    print(f"Python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"Plataforma  : {platform.platform()}")
    print(f"CPU cores   : {os.cpu_count()}")
    ram = total_ram_gb()
    print(f"RAM total   : {ram:.1f} GB" if ram else "RAM total   : desconocida")

    critical_ok = True
    py_ok = sys.version_info[:2] in [(3, 10), (3, 11), (3, 12), (3, 13)]
    critical_ok &= check(
        "Python 3.10-3.13",
        py_ok,
        f"{sys.version.split()[0]}"
        + ("" if py_ok else " (TensorFlow no soporta 3.14+)"),
        critical=True,
    )

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        print(f"TensorFlow  : {tf.__version__}")
        print(f"GPUs TF     : {gpus if gpus else 'ninguna (se usara CPU)'}")
        if not gpus and platform.system() == "Windows":
            print(
                f"{WARN}   GPU AMD: TensorFlow en Windows no soporta GPU AMD; "
                "el entrenamiento usara CPU."
            )
        tf_ok = True
    except Exception as exc:  # noqa: BLE001
        tf_ok = False
        critical_ok &= check(
            "TensorFlow", False, f"no disponible ({exc})", critical=True
        )

    for module in ("numpy", "pandas", "matplotlib", "sklearn", "PIL", "yaml"):
        try:
            __import__(module)
            print(f"{OK:6s} modulo {module}")
        except ImportError:
            critical_ok &= check(
                f"modulo {module}", False, "no instalado", critical=True
            )

    print("-" * 68)
    cfg = load_config(args.config)
    extra = list(cfg["data"].get("extra_roots") or []) + list(args.data_root)
    roots = resolve_data_roots(extra)
    print(f"Raices de datos: {[str(r) for r in roots]}")

    from src.data_loader import detect_columns, read_csv, resolve_image_path

    for label in ("csv_train", "csv_val", "csv_test"):
        csv_path = resolve_path(cfg["data"][label], PROJECT_ROOT)
        exists = csv_path.exists()
        check(f"{label}", exists, str(csv_path), critical=True)
        if not exists:
            critical_ok = False
            continue
        df = read_csv(csv_path)
        path_col, _ = detect_columns(df)
        sample = df.head(max(1, args.sample))
        found = sum(
            1
            for raw in sample[path_col]
            if resolve_image_path(str(raw), roots).exists()
        )
        ratio = found / max(1, len(sample))
        check(
            f"{label} imagenes",
            ratio == 1.0,
            f"{found}/{len(sample)} muestras localizadas ({len(df)} filas)",
            critical=ratio < 0.5,
        )
        if ratio < 0.5:
            critical_ok = False

    usage = shutil.disk_usage(HU07_ROOT)
    free_gb = usage.free / (1024**3)
    check("Espacio libre", free_gb > 2.0, f"{free_gb:.1f} GB", critical=False)

    cache_dir = HU07_ROOT / "cache"
    cache_files = list(cache_dir.glob("*.npy")) if cache_dir.exists() else []
    print(f"{OK:6s} cache: {len(cache_files)} archivos .npy en {cache_dir}")

    print("-" * 68)
    print(f"Clases definidas: {len(CLASS_NAMES)}")
    if tf_ok and py_ok and critical_ok:
        print(f"\n{OK} Entorno listo para entrenar HU07.")
        return 0
    print(f"\n{FAIL} Hay problemas criticos. Revisa los mensajes anteriores.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
