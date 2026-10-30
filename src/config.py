"""Configuracion centralizada de HU07.

HU07 es autocontenido: no importa nada de ``src/`` del proyecto original.
Solo referencia los datos mediante rutas configurables.
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── Raices ─────────────────────────────────────────────────────
HU07_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = HU07_ROOT.parent

CONFIG_DIR = HU07_ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "base.yaml"
DEFAULT_SPACE = CONFIG_DIR / "search_space.yaml"

EXPERIMENTS_DIR = HU07_ROOT / "experiments"
RESULTS_DIR = HU07_ROOT / "results"
FINAL_DIR = HU07_ROOT / "final"
DOCS_DIR = HU07_ROOT / "docs"
LOGS_DIR = HU07_ROOT / "logs"
CACHE_DIR = HU07_ROOT / "cache"
MANIFESTS_DIR = HU07_ROOT / "data_manifests"

# ── Nombres oficiales de las 43 clases GTSRB ──────────────────
CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing veh over 3.5t",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing veh > 3.5t",
}

# ── Utilidades de configuracion ────────────────────────────────
def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuracion: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"El archivo {path} no contiene un diccionario YAML.")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """Fusiona recursivamente ``override`` sobre ``base`` (sin mutar)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict:
    """Carga la configuracion base y la valida minimamente."""
    cfg = load_yaml(path or DEFAULT_CONFIG)
    cfg.setdefault("data", {})
    return cfg


def _parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Aplica overrides tipo ``clave=valor`` (soporta claves anidadas)."""
    result = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override invalido: {item!r}. Usa clave=valor.")
        key, raw = item.split("=", 1)
        keys = [k for k in key.split(".") if k]
        if not keys:
            raise ValueError(f"Override invalido: {item!r}")
        node = result
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = _parse_value(raw)
    return result


def config_hash(cfg: dict) -> str:
    """Hash corto y estable de una configuracion (para trazabilidad)."""
    import hashlib

    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:10]


# ── Rutas ──────────────────────────────────────────────────────
def resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def resolve_data_roots(extra_roots: list[str] | None = None) -> list[Path]:
    """Raices donde buscar imagenes.

    Orden:
      1. ``PROJECT_ROOT/data`` (contiene ``aumentadas/`` y ``GTSRB/``)
      2. Variables de entorno ``HU07_DATA_ROOT`` (separadas por ``;``)
      3. Raices pasadas por CLI/config (absolutas o relativas al proyecto).
    """
    roots: list[Path] = [PROJECT_ROOT / "data"]

    env = os.environ.get("HU07_DATA_ROOT", "")
    for part in env.split(os.pathsep):
        part = part.strip()
        if part:
            roots.append(resolve_path(part))

    for raw in extra_roots or []:
        roots.append(resolve_path(raw))

    unique: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r).lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def ensure_dirs() -> None:
    for d in (
        EXPERIMENTS_DIR,
        RESULTS_DIR,
        FINAL_DIR,
        DOCS_DIR,
        LOGS_DIR,
        CACHE_DIR,
        MANIFESTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(
    seed: int,
    deterministic: bool = False,
    intra_threads: int | None = None,
    inter_threads: int | None = None,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
        try:
            tf.config.threading.set_intra_op_parallelism_threads(
                max(1, int(intra_threads or os.cpu_count() or 1))
            )
            tf.config.threading.set_inter_op_parallelism_threads(
                max(1, int(inter_threads or 2))
            )
        except RuntimeError:
            pass
    except ImportError:
        pass
