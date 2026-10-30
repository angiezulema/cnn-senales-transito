"""Cliente HTTP y utilidades de imagen para la interfaz HU11.

Este módulo no depende de Streamlit: concentra la validación de imágenes y las
llamadas a la API de HU10 para poder probarlo de forma aislada.
"""

from __future__ import annotations

import io
from typing import Any

import requests
from PIL import Image, UnidentifiedImageError

DEFAULT_TIMEOUT = 30.0
HEALTH_TIMEOUT = 5.0
MAX_TOP_K = 43

_MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "BMP": "image/bmp",
    "GIF": "image/gif",
    "TIFF": "image/tiff",
}


class APIError(Exception):
    """Error de comunicación o de contrato con la API de HU10."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def normalize_api_url(api_url: str) -> str:
    """Elimina espacios y barras finales de la URL base."""
    normalized = (api_url or "").strip().rstrip("/")
    if not normalized:
        raise APIError("Configura la URL de la API.")
    return normalized


def validate_image(content: bytes) -> tuple[Image.Image, str]:
    """Valida los bytes de una imagen y devuelve (imagen RGB, MIME).

    Lanza ``ValueError`` con un mensaje apto para el usuario si el archivo está
    vacío, truncado o no es una imagen soportada por Pillow.
    """
    if not content:
        raise ValueError("El archivo está vacío.")

    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("El archivo no es una imagen válida o está dañado.") from exc

    if image.width <= 0 or image.height <= 0:
        raise ValueError("La imagen no tiene dimensiones válidas.")

    detected_format = (image.format or "").upper()
    mime = _MIME_BY_FORMAT.get(detected_format, "application/octet-stream")

    try:
        rgb_image = image.convert("RGB")
    except OSError as exc:
        raise ValueError("No se pudo convertir la imagen a RGB.") from exc

    return rgb_image, mime


def check_health(api_url: str, timeout: float = HEALTH_TIMEOUT) -> dict[str, Any]:
    """Consulta ``GET /health`` y devuelve el JSON de estado."""
    url = f"{normalize_api_url(api_url)}/health"

    try:
        response = requests.get(url, timeout=timeout)
    except requests.Timeout as exc:
        raise APIError("La API no respondió a tiempo.") from exc
    except requests.RequestException as exc:
        raise APIError(
            "No se pudo conectar con la API. Verifica que esté en ejecución."
        ) from exc

    if response.status_code != 200:
        raise APIError(
            f"La API respondió con HTTP {response.status_code}.", response.status_code
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise APIError("La API devolvió una respuesta que no es JSON.") from exc

    if not isinstance(data, dict):
        raise APIError("La API devolvió un formato de estado inesperado.")

    return data


def _extract_detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"La API respondió con HTTP {response.status_code}."
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])
    return f"La API respondió con HTTP {response.status_code}."


def predict(
    api_url: str,
    filename: str,
    content: bytes,
    top_k: int = 5,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Envía una imagen a ``POST /predict`` y normaliza la respuesta.

    Devuelve siempre un diccionario con ``class_id``, ``class_name``,
    ``confidence`` y ``top_k`` (lista de diccionarios). Si la API es una versión
    anterior sin ranking, se sintetiza ``top_k`` con la clase principal.
    """
    if not 1 <= int(top_k) <= MAX_TOP_K:
        raise ValueError(f"top_k debe estar entre 1 y {MAX_TOP_K}.")
    if not content:
        raise APIError("El archivo está vacío.", status_code=400)

    url = f"{normalize_api_url(api_url)}/predict"
    files = {
        "file": (
            filename or "imagen.jpg",
            content,
            "application/octet-stream",
        )
    }

    try:
        response = requests.post(
            url,
            files=files,
            params={"top_k": int(top_k)},
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise APIError("La API tardó demasiado en responder.") from exc
    except requests.RequestException as exc:
        raise APIError(
            "No se pudo conectar con la API. Verifica que esté en ejecución."
        ) from exc

    if response.status_code != 200:
        raise APIError(_extract_detail(response), response.status_code)

    try:
        data = response.json()
    except ValueError as exc:
        raise APIError("La API devolvió una respuesta que no es JSON.") from exc

    if not isinstance(data, dict):
        raise APIError("La API devolvió un formato de predicción inesperado.")

    try:
        class_id = int(data["class_id"])
        class_name = str(data.get("class_name") or f"Clase {class_id}")
        confidence = float(data["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise APIError("La respuesta de la API está incompleta.") from exc

    ranking: list[dict[str, Any]] = []
    raw_top_k = data.get("top_k")
    if isinstance(raw_top_k, list):
        for item in raw_top_k[: int(top_k)]:
            if not isinstance(item, dict):
                continue
            try:
                item_id = int(item["class_id"])
                item_name = str(item.get("class_name") or f"Clase {item_id}")
                item_probability = float(item["probability"])
            except (KeyError, TypeError, ValueError):
                continue
            ranking.append(
                {
                    "class_id": item_id,
                    "class_name": item_name,
                    "probability": item_probability,
                }
            )

    if not ranking:
        ranking = [
            {
                "class_id": class_id,
                "class_name": class_name,
                "probability": confidence,
            }
        ]

    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "top_k": ranking,
    }
