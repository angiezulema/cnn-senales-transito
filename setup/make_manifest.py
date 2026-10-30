"""Copia los CSV de particiones y genera un manifiesto con hashes.

Uso:
    python setup/make_manifest.py --data-root C:/Users/angie/Downloads/HU03
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

HU07_ROOT = Path(__file__).resolve().parent.parent
if str(HU07_ROOT) not in sys.path:
    sys.path.insert(0, str(HU07_ROOT))

from src.config import (
    DEFAULT_CONFIG,
    MANIFESTS_DIR,
    PROJECT_ROOT,
    load_config,
    resolve_data_roots,
    resolve_path,
)
from src.data_loader import detect_columns, read_csv, resolve_image_path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manifiesto de datos HU07")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root", action="append", default=[])
    parser.add_argument("--sample", type=int, default=100)
    args = parser.parse_args(argv)

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    extra = list(cfg["data"].get("extra_roots") or []) + list(args.data_root)
    roots = resolve_data_roots(extra)

    entries = []
    for label in ("csv_train", "csv_val", "csv_test"):
        src = resolve_path(cfg["data"][label], PROJECT_ROOT)
        if not src.exists():
            print(f"[WARN] no existe {src}")
            continue
        dst = MANIFESTS_DIR / src.name
        shutil.copyfile(src, dst)

        df = read_csv(src)
        path_col, label_col = detect_columns(df)
        sample = df.head(max(1, args.sample))
        found = sum(
            1
            for raw in sample[path_col]
            if resolve_image_path(str(raw), roots).exists()
        )
        class_counts = df[label_col].value_counts().sort_index()
        entries.append(
            {
                "partition": label,
                "source": str(src),
                "manifest": str(dst),
                "rows": int(len(df)),
                "sha256": sha256(dst),
                "sample_images_found": f"{found}/{len(sample)}",
                "classes": int(class_counts.size),
                "min_per_class": int(class_counts.min()),
                "max_per_class": int(class_counts.max()),
            }
        )
        print(f"[ok] {src.name}: {len(df)} filas, {found}/{len(sample)} imagenes")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "data_roots": [str(r) for r in roots],
        "note": "Solo se copian los CSV; las imagenes permanecen en su ubicacion.",
        "files": entries,
    }
    out = MANIFESTS_DIR / "manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifiesto: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
