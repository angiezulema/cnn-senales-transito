"""Carga, cacheo y preprocesamiento del dataset GTSRB para HU07.

Soporta multiples raices de datos para localizar imagenes:
  - ``data/aumentadas/...`` (aumentadas locales)
  - ``<root>/archive/Train/...`` (GTSRB original)
  - ``<root>/GTSRB/Train/...``
  - ``<root>/Train/...``

Las imagenes se cachean como ``uint8`` (32x32x3) en ``HU07/cache/`` para
evitar recargar y redimensionar en cada experimento de la busqueda.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import CACHE_DIR, CLASS_NAMES, PROJECT_ROOT

_PATH_COLS = ["Path", "path", "Filename", "filename", "image_path", "filepath"]
_LABEL_COLS = ["ClassId", "classid", "class_id", "label", "Label"]


# ── Lectura de CSV ─────────────────────────────────────────────
def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    path_col = next((c for c in _PATH_COLS if c in df.columns), None)
    label_col = next((c for c in _LABEL_COLS if c in df.columns), None)
    if path_col is None:
        raise ValueError(
            f"No se encontro columna de ruta. Columnas: {list(df.columns)}. "
            f"Se esperaba una de: {_PATH_COLS}"
        )
    if label_col is None:
        raise ValueError(
            f"No se encontro columna de etiqueta. Columnas: {list(df.columns)}. "
            f"Se esperaba una de: {_LABEL_COLS}"
        )
    return path_col, label_col


def read_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
    for sep in [",", ";"]:
        df = pd.read_csv(csv_path, sep=sep)
        if len(df.columns) > 1:
            return df
    return pd.read_csv(csv_path)


# ── Resolucion de rutas de imagen ──────────────────────────────
def resolve_image_path(raw_path: str, roots: list[Path]) -> Path:
    """Devuelve la primera ruta existente entre las candidatas."""
    raw = str(raw_path).replace("\\", "/")
    direct = Path(raw)
    if direct.is_absolute() and direct.exists():
        return direct

    p = Path(raw)
    parts = p.parts
    candidates: list[Path] = []

    # Aumentadas locales del proyecto original
    if parts and parts[0].lower() == "aumentadas":
        candidates.append(PROJECT_ROOT / "data" / p)

    rel = Path(*parts[1:]) if parts and parts[0].lower() == "gtsrb" else p

    for root in roots:
        candidates.extend(
            [
                root / rel,
                root / "archive" / rel,
                root / "GTSRB" / rel,
                root / p,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else direct


def sample_images_found(
    csv_path: str | Path, roots: list[Path], n: int = 100
) -> tuple[int, int]:
    """Cuenta cuantas de las primeras ``n`` filas tienen imagen existente."""
    df = read_csv(csv_path)
    path_col, _ = detect_columns(df)
    subset = df.head(n)
    found = sum(
        1
        for raw in subset[path_col]
        if resolve_image_path(str(raw), roots).exists()
    )
    return found, len(subset)


# ── Cache ──────────────────────────────────────────────────────
def _cache_key(
    csv_path: Path, roots: list[Path], height: int, width: int, limit: int | None
) -> str:
    h = hashlib.md5()
    with open(csv_path, "rb") as f:
        h.update(f.read())
    h.update("|".join(str(r).lower() for r in roots).encode("utf-8"))
    h.update(f"|{height}x{width}|uint8|limit={limit}".encode("utf-8"))
    return f"{csv_path.stem}_{h.hexdigest()[:12]}"


def _load_image(path: Path, height: int, width: int) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
    except Exception as exc:  # pragma: no cover - depende del entorno
        print(f"[WARN] No se pudo cargar {path}: {exc}", file=sys.stderr)
        return None


# ── Particiones ────────────────────────────────────────────────
def load_partition(
    csv_path: str | Path,
    roots: list[Path],
    height: int = 32,
    width: int = 32,
    num_classes: int = 43,
    use_cache: bool = True,
    limit: int | None = None,
    seed: int = 42,
    cache_dir: Path = CACHE_DIR,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Carga una particion como arrays uint8 y etiquetas int32."""
    csv_path = Path(csv_path)
    limit = int(limit) if limit else None

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(csv_path, roots, height, width, limit)
    images_file = cache_dir / f"{key}_images.npy"
    labels_file = cache_dir / f"{key}_labels.npy"

    if use_cache and images_file.exists() and labels_file.exists():
        images = np.load(images_file)
        labels = np.load(labels_file)
        if verbose:
            print(f"[cache] {csv_path.name}: {len(images)} imagenes ({key})")
        return images, labels

    df = read_csv(csv_path)
    path_col, label_col = detect_columns(df)

    if limit and limit < len(df):
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)

    images_list: list[np.ndarray] = []
    labels_list: list[int] = []
    skipped = 0

    for _, row in df.iterrows():
        label = int(row[label_col])
        if label < 0 or label >= num_classes:
            skipped += 1
            continue
        img_path = resolve_image_path(str(row[path_col]), roots)
        img = _load_image(img_path, height, width)
        if img is None:
            skipped += 1
            continue
        images_list.append(img)
        labels_list.append(label)

    if not images_list:
        preview = resolve_image_path(str(df.iloc[0][path_col]), roots)
        raise RuntimeError(
            f"No se cargo ninguna imagen de {csv_path}.\n"
            f"Raices buscadas: {[str(r) for r in roots]}\n"
            f"Primera ruta resuelta (no existe): {preview}\n"
            "Pasa --data-root apuntando a la carpeta que contiene 'archive' "
            "o 'GTSRB'."
        )

    images = np.stack(images_list).astype(np.uint8)
    labels = np.asarray(labels_list, dtype=np.int32)

    if use_cache:
        np.save(images_file, images)
        np.save(labels_file, labels)

    if verbose:
        print(
            f"[load] {csv_path.name}: {len(images)} imagenes "
            f"({skipped} omitidas) -> {key}"
        )
    return images, labels


def make_tf_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = 42,
):
    """Crea tf.data.Dataset normalizando uint8 -> float32 [0, 1]."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(images),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    ds = ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(labels: np.ndarray, num_classes: int = 43) -> dict:
    """Pesos balanceados; las clases ausentes reciben peso 1.0.

    Con el dataset completo estan las 43 clases. En subconjuntos (smoke test)
    pueden faltar clases y ``compute_class_weight`` fallaria si se le pasan
    todas; por eso solo se calculan las presentes.
    """
    from sklearn.utils.class_weight import compute_class_weight

    present = np.unique(labels)
    if len(present) == 0:
        return {i: 1.0 for i in range(num_classes)}

    weights = compute_class_weight("balanced", classes=present, y=labels)
    weight_map = {int(i): 1.0 for i in range(num_classes)}
    weight_map.update({int(c): float(w) for c, w in zip(present, weights)})
    return weight_map


def dataset_summary(labels: np.ndarray) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    return {
        "total": int(len(labels)),
        "classes_present": int(len(unique)),
        "min_per_class": int(counts.min()) if len(counts) else 0,
        "max_per_class": int(counts.max()) if len(counts) else 0,
        "class_distribution": {
            CLASS_NAMES.get(int(c), str(c)): int(n)
            for c, n in zip(unique, counts)
        },
    }
