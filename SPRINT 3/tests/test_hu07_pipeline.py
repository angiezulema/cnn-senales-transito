"""Tests del pipeline HU07 (config, datos, modelo y busqueda)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

HU07_ROOT = Path(__file__).resolve().parent.parent
if str(HU07_ROOT) not in sys.path:
    sys.path.insert(0, str(HU07_ROOT))

from src.config import (
    CLASS_NAMES,
    DEFAULT_CONFIG,
    apply_overrides,
    config_hash,
    load_config,
)
from src.data_loader import load_partition, make_tf_dataset, resolve_image_path


def _has_tf() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


class TestConfig:
    def test_load_default(self):
        cfg = load_config(DEFAULT_CONFIG)
        assert cfg["num_classes"] == 43
        assert cfg["img_height"] == 32
        assert "data" in cfg

    def test_class_names_complete(self):
        assert len(CLASS_NAMES) == 43
        assert set(CLASS_NAMES) == set(range(43))

    def test_apply_overrides_top_level(self):
        cfg = load_config(DEFAULT_CONFIG)
        out = apply_overrides(cfg, ["learning_rate=0.0005", "batch_size=128"])
        assert out["learning_rate"] == 0.0005
        assert out["batch_size"] == 128
        assert cfg["learning_rate"] != 0.0005

    def test_apply_overrides_nested(self):
        cfg = load_config(DEFAULT_CONFIG)
        out = apply_overrides(cfg, ["early_stopping.patience=3"])
        assert out["early_stopping"]["patience"] == 3

    def test_apply_overrides_invalid(self):
        cfg = load_config(DEFAULT_CONFIG)
        with pytest.raises(ValueError):
            apply_overrides(cfg, ["sin_igual"])

    def test_config_hash_stable(self):
        cfg = load_config(DEFAULT_CONFIG)
        assert config_hash(cfg) == config_hash(load_config(DEFAULT_CONFIG))
        assert config_hash(cfg) != config_hash(apply_overrides(cfg, ["batch_size=32"]))


class TestPathResolution:
    def test_archive_root(self, tmp_path: Path):
        target = tmp_path / "archive" / "Train" / "1" / "a.png"
        target.parent.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(target)

        resolved = resolve_image_path("GTSRB/Train/1/a.png", [tmp_path])
        assert resolved == target

    def test_direct_root(self, tmp_path: Path):
        target = tmp_path / "Train" / "2" / "b.png"
        target.parent.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(target)

        resolved = resolve_image_path("Train/2/b.png", [tmp_path])
        assert resolved == target


class TestCacheAndLoader:
    def test_load_and_cache(self, tmp_path: Path):
        root = tmp_path / "data_root"
        img_dir = root / "archive" / "Train"
        rows = []
        rng = np.random.default_rng(0)
        for class_id in (1, 2):
            d = img_dir / str(class_id)
            d.mkdir(parents=True)
            for i in range(3):
                name = f"{class_id:05d}_{i:05d}.png"
                arr = rng.integers(0, 255, (30, 30, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / name)
                rows.append(
                    {"Path": f"GTSRB/Train/{class_id}/{name}", "ClassId": class_id}
                )

        csv_path = tmp_path / "train.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        cache_dir = tmp_path / "cache"

        images, labels = load_partition(
            csv_path,
            [root],
            height=32,
            width=32,
            use_cache=True,
            cache_dir=cache_dir,
        )
        assert images.shape == (6, 32, 32, 3)
        assert images.dtype == np.uint8
        assert labels.dtype == np.int32
        assert set(np.unique(labels)) == {1, 2}
        assert len(list(cache_dir.glob("*.npy"))) == 2

        images2, labels2 = load_partition(
            csv_path,
            [root],
            height=32,
            width=32,
            use_cache=True,
            cache_dir=cache_dir,
        )
        np.testing.assert_array_equal(images, images2)
        np.testing.assert_array_equal(labels, labels2)

    def test_limit_is_respected(self, tmp_path: Path):
        root = tmp_path / "data_root"
        d = root / "archive" / "Train" / "3"
        d.mkdir(parents=True)
        rows = []
        for i in range(5):
            name = f"00003_{i:05d}.png"
            Image.fromarray(
                np.zeros((10, 10, 3), dtype=np.uint8)
            ).save(d / name)
            rows.append({"Path": f"GTSRB/Train/3/{name}", "ClassId": 3})
        csv_path = tmp_path / "train.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        images, labels = load_partition(
            csv_path,
            [root],
            use_cache=False,
            limit=2,
            cache_dir=tmp_path / "cache",
        )
        assert len(images) == len(labels) == 2


@pytest.mark.skipif(not _has_tf(), reason="TensorFlow no instalado")
class TestModel:
    def _cfg(self) -> dict:
        cfg = load_config(DEFAULT_CONFIG)
        cfg.update(
            {
                "conv_filters": [8, 16],
                "dense_units": 32,
                "img_height": 32,
                "img_width": 32,
                "img_channels": 3,
                "num_classes": 43,
            }
        )
        return cfg

    def test_build_and_compile(self):
        from src.model import build_cnn, compile_model

        model = build_cnn(self._cfg())
        compile_model(model, self._cfg())
        assert model.input_shape == (None, 32, 32, 3)
        assert model.output_shape == (None, 43)
        assert model.count_params() > 0

    def test_label_smoothing_loss(self):
        import tensorflow as tf

        from src.model import SparseCategoricalCrossentropyLS

        y_true = tf.constant([0, 2], dtype=tf.int32)
        probs = tf.constant(
            [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], dtype=tf.float32
        )
        loss_plain = SparseCategoricalCrossentropyLS(0.0)(y_true, probs)
        expected = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                y_true, probs, from_logits=False
            )
        )
        np.testing.assert_allclose(
            loss_plain.numpy(), expected.numpy(), rtol=1e-5
        )

        loss_smooth = SparseCategoricalCrossentropyLS(0.1)(y_true, probs)
        assert float(loss_smooth) > float(loss_plain)

    def test_dataset_normalizes(self):
        images = np.full((4, 32, 32, 3), 255, dtype=np.uint8)
        labels = np.array([0, 1, 2, 3], dtype=np.int32)
        ds = make_tf_dataset(images, labels, batch_size=2)
        batch_images, batch_labels = next(iter(ds))
        assert batch_images.shape == (2, 32, 32, 3)
        np.testing.assert_allclose(batch_images.numpy().max(), 1.0)
        assert batch_labels.shape == (2,)


@pytest.mark.skipif(not _has_tf(), reason="TensorFlow no instalado")
class TestTrialGeneration:
    def test_trials_unique_and_count(self):
        from src.run_search import generate_trials

        cfg = load_config(DEFAULT_CONFIG)
        space = {
            "learning_rate": [0.001, 0.0005],
            "batch_size": [32, 64],
            "optimizer": ["adam", "adamw"],
            "weight_decay": [0.0001],
        }
        trials = generate_trials(cfg, space, 6, seed=1, include_baseline=True)
        assert len(trials) == 6
        signatures = {
            (
                t["learning_rate"],
                t["batch_size"],
                t["optimizer"],
                t["weight_decay"],
            )
            for _, t in trials
        }
        assert len(signatures) == len(trials)
        assert trials[0][0] == "E00_baseline"
        for _, trial in trials:
            if trial.get("optimizer") == "adam":
                assert trial["weight_decay"] == 0.0
