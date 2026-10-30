"""Tests del cliente HTTP de HU11 (sin Streamlit ni TensorFlow)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import requests
from PIL import Image

HU11_ROOT = Path(__file__).resolve().parent.parent
if str(HU11_ROOT) not in sys.path:
    sys.path.insert(0, str(HU11_ROOT))

import api_client  # noqa: E402


def _image_bytes(fmt: str = "PNG", mode: str = "RGB") -> bytes:
    image = Image.new(mode, (40, 30), (200, 30, 30, 255) if mode == "RGBA" else (200, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, invalid_json: bool = False):
        self.status_code = status_code
        self._payload = payload
        self._invalid_json = invalid_json

    def json(self):
        if self._invalid_json:
            raise ValueError("JSON inválido")
        return self._payload


class TestValidateImage:
    def test_accepts_png_and_returns_rgb(self):
        image, mime = api_client.validate_image(_image_bytes("PNG"))
        assert image.mode == "RGB"
        assert image.size == (40, 30)
        assert mime == "image/png"

    def test_converts_rgba_to_rgb(self):
        image, mime = api_client.validate_image(_image_bytes("PNG", mode="RGBA"))
        assert image.mode == "RGB"
        assert mime == "image/png"

    def test_rejects_empty_file(self):
        with pytest.raises(ValueError, match="vacío"):
            api_client.validate_image(b"")

    def test_rejects_non_image(self):
        with pytest.raises(ValueError, match="no es una imagen"):
            api_client.validate_image(b"esto no es una imagen")


class TestHealth:
    def test_check_health_ok(self, monkeypatch):
        captured = {}

        def fake_get(url, timeout):
            captured["url"] = url
            captured["timeout"] = timeout
            return FakeResponse(payload={"status": "ok", "model_loaded": True})

        monkeypatch.setattr(api_client.requests, "get", fake_get)
        data = api_client.check_health("http://localhost:8000/")
        assert data["model_loaded"] is True
        assert captured["url"] == "http://localhost:8000/health"

    def test_check_health_connection_error(self, monkeypatch):
        def fake_get(url, timeout):
            raise requests.ConnectionError("sin conexión")

        monkeypatch.setattr(api_client.requests, "get", fake_get)
        with pytest.raises(api_client.APIError, match="No se pudo conectar"):
            api_client.check_health("http://localhost:8000")

    def test_check_health_http_error(self, monkeypatch):
        monkeypatch.setattr(
            api_client.requests,
            "get",
            lambda url, timeout: FakeResponse(status_code=503, payload={"detail": "x"}),
        )
        with pytest.raises(api_client.APIError) as exc_info:
            api_client.check_health("http://localhost:8000")
        assert exc_info.value.status_code == 503


class TestPredict:
    def test_predict_ok_with_top_k(self, monkeypatch):
        captured = {}

        def fake_post(url, files, params, timeout):
            captured["url"] = url
            captured["files"] = files
            captured["params"] = params
            return FakeResponse(
                payload={
                    "class_id": 14,
                    "class_name": "Stop",
                    "confidence": 0.9987,
                    "top_k": [
                        {"class_id": 14, "class_name": "Stop", "probability": 0.9987},
                        {"class_id": 17, "class_name": "No entry", "probability": 0.0006},
                    ],
                }
            )

        monkeypatch.setattr(api_client.requests, "post", fake_post)
        result = api_client.predict(
            "http://localhost:8000", "stop.jpg", _image_bytes(), top_k=2
        )
        assert result["class_id"] == 14
        assert result["class_name"] == "Stop"
        assert len(result["top_k"]) == 2
        assert captured["url"] == "http://localhost:8000/predict"
        assert captured["params"] == {"top_k": 2}

    def test_predict_old_api_synthesizes_top_k(self, monkeypatch):
        monkeypatch.setattr(
            api_client.requests,
            "post",
            lambda url, files, params, timeout: FakeResponse(
                payload={"class_id": 1, "class_name": "30", "confidence": 0.5}
            ),
        )
        result = api_client.predict("http://localhost:8000", "x.png", b"abc")
        assert result["top_k"] == [
            {"class_id": 1, "class_name": "30", "probability": 0.5}
        ]

    def test_predict_http_error_uses_detail(self, monkeypatch):
        monkeypatch.setattr(
            api_client.requests,
            "post",
            lambda url, files, params, timeout: FakeResponse(
                status_code=400, payload={"detail": "El archivo está vacío."}
            ),
        )
        with pytest.raises(api_client.APIError, match="El archivo está vacío") as exc_info:
            api_client.predict("http://localhost:8000", "x.png", b"abc")
        assert exc_info.value.status_code == 400

    def test_predict_timeout(self, monkeypatch):
        def fake_post(url, files, params, timeout):
            raise requests.Timeout("timeout")

        monkeypatch.setattr(api_client.requests, "post", fake_post)
        with pytest.raises(api_client.APIError, match="tardó demasiado"):
            api_client.predict("http://localhost:8000", "x.png", b"abc")

    def test_predict_rejects_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            api_client.predict("http://localhost:8000", "x.png", b"abc", top_k=0)

    def test_predict_rejects_empty_content(self):
        with pytest.raises(api_client.APIError, match="vacío") as exc_info:
            api_client.predict("http://localhost:8000", "x.png", b"")
        assert exc_info.value.status_code == 400
