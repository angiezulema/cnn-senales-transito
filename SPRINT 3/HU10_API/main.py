"""HU10 - API FastAPI para clasificación de señales de tráfico (GTSRB).

Sirve el modelo exportado por HU07 sin modificar sus artefactos. El modelo se
carga en el evento ``startup`` y se expone mediante los endpoints ``/health`` y
``/predict``.

Origen de rutas y parámetros (todo en modo lectura):
- Modelo: ``final/model_final.keras`` (ver ``final/export_report.json``).
- Nombres de clase: ``final/class_names.json`` (43 clases, claves "0".."42").
- Preprocesamiento: ``final/preprocessing.json`` (RGB, 32x32, bilineal,
  scale = 1/255, NHWC, float32) y ``config/base.yaml``
  (``img_height: 32``, ``img_width: 32``, ``img_channels: 3``).
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hu10.api")

# HU10_API/main.py -> padre = raíz del proyecto (HU07, solo lectura)
HU10_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HU10_DIR.parent

# Artefactos finales identificados en el Paso 1 de exploración
FINAL_DIR = PROJECT_ROOT / "final"
MODEL_PATH = FINAL_DIR / "model_final.keras"
CLASS_NAMES_PATH = FINAL_DIR / "class_names.json"
PREPROCESSING_PATH = FINAL_DIR / "preprocessing.json"

# Fallback defensivo si preprocessing.json no estuviera disponible.
# Valores tomados de config/base.yaml (líneas 7-9) y final/preprocessing.json.
DEFAULT_IMG_HEIGHT = 32
DEFAULT_IMG_WIDTH = 32
DEFAULT_SCALE = 1.0 / 255.0
DEFAULT_RESIZE_METHOD = "bilinear"

# Estado global cargado en startup
model: tf.keras.Model | None = None
class_names: dict[str, str] = {}
preprocessing: dict = {}

app = FastAPI(
    title="HU10 - Traffic Sign Classifier API",
    description="Clasificación de 43 señales de tráfico (GTSRB) con una CNN de HU07.",
    version="1.0.0",
)

# CORS abierto para permitir el consumo desde un frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_artifacts() -> None:
    """Carga los JSON de clases y preprocesamiento exportados por HU07."""
    global class_names, preprocessing

    with open(CLASS_NAMES_PATH, encoding="utf-8") as fh:
        class_names = json.load(fh)

    with open(PREPROCESSING_PATH, encoding="utf-8") as fh:
        preprocessing = json.load(fh)

    logger.info(
        "Metadatos cargados: %d clases, entrada %sx%s, scale=%s",
        len(class_names),
        preprocessing.get("img_width", DEFAULT_IMG_WIDTH),
        preprocessing.get("img_height", DEFAULT_IMG_HEIGHT),
        preprocessing.get("scale", DEFAULT_SCALE),
    )


@app.on_event("startup")
def load_model() -> None:
    """Carga el modelo Keras al iniciar la API.

    Se usa ``compile=False`` porque el modelo de HU07 fue compilado con la loss
    personalizada ``SparseCategoricalCrossentropyLS`` (src/model.py), que no es
    necesaria para inferencia y no está disponible en este módulo.
    """
    global model

    try:
        _load_artifacts()

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")

        logger.info("Cargando modelo desde %s", MODEL_PATH)
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        logger.info("Modelo cargado correctamente")
    except Exception:
        model = None
        logger.exception("No se pudo cargar el modelo o los metadatos")


@app.get("/health")
def health() -> dict:
    """Estado básico del servicio y del modelo."""
    return {"status": "ok", "model_loaded": model is not None}


def _preprocess_image(raw: bytes) -> np.ndarray:
    """Valida y preprocesa la imagen según el contrato de HU07.

    Pasos: decodificar con PIL, convertir a RGB, redimensionar a 32x32 con
    interpolación bilineal (final/preprocessing.json), convertir a float32 y
    escalar a [0, 1] (scale = 1/255). Devuelve un tensor NHWC (1, 32, 32, 3).
    """
    try:
        # ``load()`` fuerza la decodificación y detecta archivos truncados.
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="El archivo no es una imagen válida."
        ) from exc

    try:
        image = image.convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="No se pudo convertir la imagen a RGB."
        ) from exc

    height = int(preprocessing.get("img_height", DEFAULT_IMG_HEIGHT))
    width = int(preprocessing.get("img_width", DEFAULT_IMG_WIDTH))
    scale = float(preprocessing.get("scale", DEFAULT_SCALE))

    # El modelo se exportó con resize bilineal (final/preprocessing.json).
    if preprocessing.get("resize_method", DEFAULT_RESIZE_METHOD) != "bilinear":
        logger.warning(
            "resize_method distinto de 'bilinear'; se usará bilineal por compatibilidad"
        )
    resized = image.resize((width, height), Image.Resampling.BILINEAR)

    array = np.asarray(resized, dtype=np.float32) * scale
    return np.expand_dims(array, axis=0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Clasifica una imagen y devuelve clase, nombre y confianza."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Consulte /health.",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    tensor = _preprocess_image(raw)

    try:
        probabilities = model.predict(tensor, verbose=0)[0]
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
    except Exception as exc:
        logger.exception("Error durante la inferencia")
        raise HTTPException(
            status_code=500, detail="Error interno durante la predicción."
        ) from exc

    return {
        "class_id": class_id,
        "class_name": class_names.get(str(class_id), f"Clase {class_id}"),
        "confidence": round(confidence, 4),
    }
