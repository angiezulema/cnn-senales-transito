"""Arquitectura CNN configurable para GTSRB (43 clases).

La arquitectura base replica HU06:
    3 bloques convolucionales (32→64→128) + clasificador denso.
Los hiperparametros (filtros, dropout, unidades densas) se leen del dict
de configuracion para permitir la busqueda de HU07.
"""

from __future__ import annotations

import keras as _keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@_keras.saving.register_keras_serializable(package="HU07")
class SparseCategoricalCrossentropyLS(keras.losses.Loss):
    """Sparse categorical crossentropy con label smoothing opcional."""

    def __init__(
        self,
        smoothing: float = 0.0,
        name: str = "sparse_ce_ls",
        reduction: str = "sum_over_batch_size",
    ):
        super().__init__(name=name, reduction=reduction)
        self.smoothing = float(smoothing)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["smoothing"] = self.smoothing
        return cfg

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0)
        log_p = tf.math.log(y_pred)
        n_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        smooth = self.smoothing / n_classes

        indices = tf.stack([tf.range(tf.shape(y_true)[0]), y_true], axis=1)
        true_log_p = tf.gather_nd(log_p, indices)
        loss = -(
            (1.0 - self.smoothing) * true_log_p
            + smooth * tf.reduce_sum(log_p, axis=-1)
        )
        return loss


def load_trained_model(path: str) -> keras.Model:
    """Carga un modelo guardado resolviendo la loss personalizada."""
    return keras.models.load_model(
        str(path),
        custom_objects={
            "SparseCategoricalCrossentropyLS": SparseCategoricalCrossentropyLS,
            "HU07>SparseCategoricalCrossentropyLS": SparseCategoricalCrossentropyLS,
        },
    )


def build_cnn(cfg: dict) -> keras.Model:
    """Construye la CNN segun la configuracion dada."""
    height = int(cfg.get("img_height", 32))
    width = int(cfg.get("img_width", 32))
    channels = int(cfg.get("img_channels", 3))
    num_classes = int(cfg.get("num_classes", 43))

    filters = [int(f) for f in cfg.get("conv_filters", [32, 64, 128])]
    dense_units = int(cfg.get("dense_units", 512))
    dropout_conv = float(cfg.get("dropout_conv", 0.25))
    dropout_dense = float(cfg.get("dropout_dense", 0.5))

    inputs = layers.Input(shape=(height, width, channels), name="input_image")
    x = inputs

    for block_idx, n_filters in enumerate(filters, start=1):
        for conv_idx in (1, 2):
            x = layers.Conv2D(
                n_filters,
                (3, 3),
                padding="same",
                name=f"block{block_idx}_conv{conv_idx}",
            )(x)
            x = layers.BatchNormalization(
                name=f"block{block_idx}_bn{conv_idx}"
            )(x)
            x = layers.Activation("relu", name=f"block{block_idx}_relu{conv_idx}")(x)
        x = layers.MaxPooling2D(
            pool_size=(2, 2), name=f"block{block_idx}_pool"
        )(x)
        x = layers.Dropout(dropout_conv, name=f"block{block_idx}_dropout")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, name="dense")(x)
    x = layers.BatchNormalization(name="dense_bn")(x)
    x = layers.Activation("relu", name="dense_relu")(x)
    x = layers.Dropout(dropout_dense, name="dense_dropout")(x)

    outputs = layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="traffic_sign_cnn")


def compile_model(model: keras.Model, cfg: dict) -> keras.Model:
    """Compila el modelo con optimizador, LR y label smoothing configurables."""
    lr = float(cfg.get("learning_rate", 1e-3))
    optimizer_name = str(cfg.get("optimizer", "adam")).lower()
    weight_decay = float(cfg.get("weight_decay", 0.0) or 0.0)
    label_smoothing = float(cfg.get("label_smoothing", 0.0) or 0.0)

    if optimizer_name == "adamw":
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    else:
        raise ValueError(f"Optimizador no soportado: {optimizer_name}")

    model.compile(
        optimizer=optimizer,
        loss=SparseCategoricalCrossentropyLS(smoothing=label_smoothing),
        metrics=["accuracy"],
        jit_compile=bool(cfg.get("jit_compile", False)),
    )
    return model
