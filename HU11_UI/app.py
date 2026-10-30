"""HU11 - Interfaz Streamlit para el reconocimiento de señales de tráfico.

La aplicación consume la API FastAPI de HU10 (``POST /predict`` y
``GET /health``) y no carga TensorFlow ni el modelo directamente. De este modo
la UI es ligera y puede desplegarse en un contenedor independiente.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any

import pandas as pd
import streamlit as st

from api_client import APIError, check_health, predict, validate_image

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "30"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
TOP_K = 5

st.set_page_config(
    page_title="Señales de tráfico | HU11",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "health": None,
        "health_error": None,
        "health_latency_ms": None,
        "result": None,
        "result_error": None,
        "last_file_hash": None,
        "last_filename": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _check_api(api_url: str) -> None:
    started = time.perf_counter()
    try:
        data = check_health(api_url)
    except APIError as exc:
        st.session_state.health = None
        st.session_state.health_error = exc.message
        st.session_state.health_latency_ms = None
    else:
        st.session_state.health = data
        st.session_state.health_error = None
        st.session_state.health_latency_ms = (time.perf_counter() - started) * 1000


def _render_health_badge() -> None:
    health = st.session_state.health
    error = st.session_state.health_error

    if error:
        st.error(error)
    elif health is None:
        st.caption("Estado de la API sin comprobar.")
    elif health.get("model_loaded"):
        st.success("API conectada y modelo cargado.")
    else:
        st.warning("La API responde, pero el modelo no está cargado.")


def _run_prediction(
    api_url: str, filename: str, content: bytes, file_hash: str
) -> None:
    try:
        result = predict(
            api_url,
            filename=filename,
            content=content,
            top_k=TOP_K,
            timeout=API_TIMEOUT,
        )
    except (APIError, ValueError) as exc:
        st.session_state.result = None
        st.session_state.result_error = str(exc)
    else:
        st.session_state.result = result
        st.session_state.result_error = None
        st.session_state.last_file_hash = file_hash
        st.session_state.last_filename = filename


def _render_result() -> None:
    result = st.session_state.result
    error = st.session_state.result_error

    if error:
        st.error(error)
        return
    if not result:
        st.info("Sube una imagen y pulsa **Analizar señal** para ver el resultado.")
        return

    confidence = max(0.0, min(1.0, float(result["confidence"])))
    st.metric("Señal detectada", result["class_name"])
    col_id, col_conf = st.columns(2)
    col_id.metric("Clase", result["class_id"])
    col_conf.metric("Confianza", f"{confidence:.2%}")
    st.progress(confidence, text=f"Probabilidad de la clase principal: {confidence:.2%}")

    ranking = result.get("top_k") or []
    if ranking:
        table = pd.DataFrame(
            [
                {
                    "Clase": item["class_id"],
                    "Nombre": item["class_name"],
                    "Probabilidad": float(item["probability"]),
                }
                for item in ranking
            ]
        )
        chart = table.set_index("Nombre")[["Probabilidad"]]
        st.bar_chart(chart, height=240)

        display = table.copy()
        display["Probabilidad"] = display["Probabilidad"].map(lambda v: f"{v:.2%}")
        st.dataframe(display, hide_index=True, width="stretch")

    st.caption(
        "El modelo fue entrenado con GTSRB (43 clases) y entradas RGB de 32x32. "
        "Una confianza baja puede indicar una imagen borrosa, con poca luz o "
        "fuera de las clases del dataset."
    )

    with st.expander("Respuesta JSON de la API"):
        st.json(result)


def _render_predict_tab(api_url: str) -> None:
    left, right = st.columns([1, 1], gap="large")

    content: bytes | None = None
    filename = "imagen.jpg"

    with left:
        st.subheader("1. Imagen de entrada")
        source = st.radio(
            "Origen",
            options=["Archivo", "Cámara"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if source == "Archivo":
            uploaded = st.file_uploader(
                "Selecciona una imagen de una señal de tráfico",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                help=f"Tamaño máximo: {MAX_UPLOAD_MB} MB.",
            )
            if uploaded is not None:
                filename = uploaded.name or filename
                content = uploaded.getvalue()
        else:
            camera = st.camera_input("Toma una foto de la señal")
            if camera is not None:
                filename = "camara.jpg"
                content = camera.getvalue()

        if content is None:
            st.caption("Formatos admitidos: JPG, PNG, WEBP y BMP.")
        elif len(content) > MAX_UPLOAD_BYTES:
            st.error(f"La imagen supera el límite de {MAX_UPLOAD_MB} MB.")
            content = None
        else:
            try:
                image, _ = validate_image(content)
            except ValueError as exc:
                st.error(str(exc))
                content = None
            else:
                st.image(image, caption=filename, width="stretch")

        file_hash = hashlib.sha256(content).hexdigest() if content else None
        if file_hash != st.session_state.last_file_hash:
            st.session_state.result = None
            st.session_state.result_error = None
            st.session_state.last_file_hash = file_hash
            st.session_state.last_filename = filename if content else None

        analyze = st.button(
            "Analizar señal",
            type="primary",
            width="stretch",
            disabled=content is None,
        )
        if analyze and content is not None:
            with st.spinner("Consultando el modelo..."):
                _run_prediction(api_url, filename, content, file_hash or "")
            st.session_state.last_file_hash = file_hash
            st.session_state.last_filename = filename

    with right:
        st.subheader("2. Resultado")
        _render_result()


def _render_info_tab() -> None:
    st.subheader("Sobre el sistema")
    st.markdown(
        """
        Esta interfaz forma parte del **Sistema Inteligente de Reconocimiento de
        Señales de Tráfico para la Asistencia Vial de Conductores**. La cadena
        completa es:

        1. **HU07**: entrenamiento y exportación de la CNN (GTSRB, 43 clases).
        2. **HU10**: API FastAPI que sirve el modelo.
        3. **HU11**: interfaz Streamlit y despliegue con Docker.
        """
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Modelo")
        st.markdown(
            """
            - Arquitectura: CNN con 3 bloques convolucionales (32/64/128).
            - Entrada: RGB, 32x32, `float32`, escala `[0, 1]`, layout NHWC.
            - Salida: 43 probabilidades softmax.
            - Accuracy en test: **98.39 %**.
            - F1 macro en test: **97.43 %**.
            - Muestras de test: **12 630**.
            """
        )
    with col_b:
        st.markdown("#### Contrato de entrada")
        st.json(
            {
                "color_mode": "RGB",
                "img_height": 32,
                "img_width": 32,
                "resize_method": "bilinear",
                "scale": 0.00392156862745098,
                "input_layout": "NHWC",
                "input_dtype": "float32",
                "num_classes": 43,
            }
        )

    st.markdown("#### Clases reconocidas")
    st.caption(
        "El modelo distingue 43 señales del dataset GTSRB: límites de velocidad, "
        "prohibiciones, advertencias, prioridades y señales de obligación."
    )


def _render_status_tab(api_url: str) -> None:
    st.subheader("Estado de la API")
    col_url, col_button = st.columns([3, 1])
    col_url.code(api_url, language=None)
    if col_button.button("Comprobar ahora", width="stretch"):
        with st.spinner("Consultando /health..."):
            _check_api(api_url)

    health = st.session_state.health
    error = st.session_state.health_error
    latency = st.session_state.health_latency_ms

    if error:
        st.error(error)
    elif health is None:
        st.info("Pulsa **Comprobar ahora** para consultar el estado del servicio.")
    else:
        col_status, col_model, col_latency = st.columns(3)
        col_status.metric("Servicio", str(health.get("status", "desconocido")))
        col_model.metric(
            "Modelo", "Cargado" if health.get("model_loaded") else "No cargado"
        )
        col_latency.metric(
            "Latencia /health",
            f"{latency:.0f} ms" if latency is not None else "—",
        )
        st.json(health)

    with st.expander("Solución de problemas"):
        st.markdown(
            """
            - **No se pudo conectar**: verifica que la API esté en ejecución y que
              la URL sea correcta (`http://127.0.0.1:8000` en local,
              `http://api:8000` dentro de Docker Compose).
            - **HTTP 503**: la API está activa, pero el modelo no se cargó. Revisa
              los logs del contenedor o la ruta `final/model_final.keras`.
            - **HTTP 400**: el archivo no es una imagen válida o está vacío.
            - **Timeout**: la primera inferencia puede tardar mientras
              TensorFlow inicializa; vuelve a intentarlo.
            """
        )

    try:
        docs_url = f"{api_url.rstrip('/')}/docs"
        st.link_button("Abrir Swagger UI de la API", docs_url)
    except Exception:
        pass


def main() -> None:
    _init_state()

    st.sidebar.title("Configuración")
    api_url = st.sidebar.text_input(
        "URL de la API",
        value=DEFAULT_API_URL,
        help="Local: http://127.0.0.1:8000 · Docker Compose: http://api:8000",
    )
    if st.sidebar.button("Comprobar conexión", width="stretch"):
        with st.spinner("Consultando /health..."):
            _check_api(api_url)

    _render_health_badge()

    st.sidebar.divider()
    st.sidebar.subheader("Modelo")
    st.sidebar.markdown(
        """
        - CNN GTSRB · 43 clases
        - Entrada 32x32 RGB
        - Accuracy test: 98.39 %
        - F1 macro: 97.43 %
        """
    )
    st.sidebar.caption("HU07 (modelo) → HU10 (API) → HU11 (UI + Docker)")

    st.title("Reconocimiento de Señales de Tráfico")
    st.caption(
        "Sistema Inteligente de Reconocimiento de Señales para la Asistencia Vial "
        "de Conductores · Interfaz Streamlit (HU11)"
    )

    tab_predict, tab_info, tab_status = st.tabs(
        ["Clasificar", "Información", "Estado de la API"]
    )
    with tab_predict:
        _render_predict_tab(api_url)
    with tab_info:
        _render_info_tab()
    with tab_status:
        _render_status_tab(api_url)


if __name__ == "__main__":
    main()
