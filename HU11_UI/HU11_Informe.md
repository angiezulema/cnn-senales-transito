# HU11 - Interfaz Streamlit y despliegue con Docker

## 1. Descripción

La HU11 implementa la interfaz de usuario del Sistema Inteligente de
Reconocimiento de Señales de Tráfico y su despliegue en contenedores. La UI está
construida con **Streamlit** y consume la API **FastAPI** de HU10 mediante HTTP,
por lo que **no carga TensorFlow ni el modelo**: el contenedor de la interfaz es
ligero e independiente del de inferencia.

Cadena completa del sistema:

```
HU07 (modelo CNN) -> HU10 (API FastAPI) -> HU11 (UI Streamlit + Docker)
```

Artefactos consumidos por la API (solo lectura):

| Artefacto | Ruta |
|---|---|
| Modelo Keras | `final/model_final.keras` |
| Nombres de clase | `final/class_names.json` |
| Contrato de entrada | `final/preprocessing.json` |

Métricas del modelo en test (`final/test_metrics.json`):

| Métrica | Valor |
|---|---|
| Accuracy | 98.39 % |
| Precision macro | 97.55 % |
| Recall macro | 97.37 % |
| F1 macro | 97.43 % |
| Muestras de test | 12 630 |

## 2. Estructura de la entrega

```
HU10_API/
├── main.py                  # API FastAPI (/health, /predict)
├── requirements.txt         # dependencias locales
├── requirements-docker.txt  # dependencias para la imagen (tensorflow-cpu)
└── Dockerfile               # imagen CPU de la API (puerto 8000)
HU11_UI/
├── app.py                   # interfaz Streamlit
├── api_client.py            # cliente HTTP + validación de imágenes (sin Streamlit)
├── requirements.txt         # dependencias de la UI
├── Dockerfile               # imagen Streamlit (puerto 8501)
├── HU11_Informe.md          # este documento
└── tests/
    └── test_api_client.py   # 13 tests del cliente y la validación
docker-compose.yml           # orquesta api + ui con healthchecks
.dockerignore                # excluye venv, datos, caché, logs y la copia duplicada
```

## 3. Interfaz Streamlit (`app.py`)

La aplicación se organiza en tres pestañas:

1. **Clasificar**: carga de imagen por archivo (JPG/PNG/WEBP/BMP) o cámara,
   vista previa, validación local (tamaño máximo y formato) y botón
   *Analizar señal*. Muestra clase detectada, confianza, barra de progreso,
   gráfico de barras con el top-5, tabla y el JSON crudo de la respuesta.
2. **Información**: descripción del sistema, arquitectura del modelo y contrato
   de entrada.
3. **Estado de la API**: consulta `GET /health`, latencia, estado del modelo,
   enlace a Swagger UI y guía de solución de problemas.

Detalles relevantes:

- La URL de la API es configurable en la barra lateral (`API_URL`).
- Si cambia la imagen cargada, se invalida el resultado anterior (hash SHA-256).
- Los errores de red, HTTP 400/503 y timeouts se muestran con mensajes claros.
- La validación de imagen usa Pillow (`api_client.validate_image`) y convierte a
  RGB antes de enviar.
- La UI no depende de Streamlit para sus pruebas: `api_client.py` es importable
  de forma aislada.

Variables de entorno de la UI:

| Variable | Por defecto | Uso |
|---|---|---|
| `API_URL` | `http://127.0.0.1:8000` | URL base de la API (en Compose: `http://api:8000`) |
| `API_TIMEOUT` | `30` | Timeout de `POST /predict` en segundos |
| `MAX_UPLOAD_MB` | `10` | Límite de subida en MB |

## 4. Contrato API consumido

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict?top_k=5`

- `multipart/form-data` con el campo `file`.
- Devuelve:

```json
{
  "class_id": 14,
  "class_name": "Stop",
  "confidence": 0.9987,
  "top_k": [
    {"class_id": 14, "class_name": "Stop", "probability": 0.9987},
    {"class_id": 17, "class_name": "No entry", "probability": 0.0006}
  ]
}
```

Errores: `400` (archivo vacío o no es imagen), `500` (fallo de inferencia),
`503` (modelo no cargado).

## 5. Docker

### 5.1 Imagen de la API (`HU10_API/Dockerfile`)

- Base `python:3.12-slim` + `libgomp1` (requerido por TensorFlow en Linux).
- Instala `HU10_API/requirements-docker.txt` (`tensorflow-cpu`, FastAPI, Uvicorn,
  Pillow, `python-multipart`, NumPy, PyYAML).
- Copia `HU10_API/main.py` y los tres artefactos de `final/`.
- Expone `8000`, define `HEALTHCHECK` contra `/health` y arranca Uvicorn.

### 5.2 Imagen de la UI (`HU11_UI/Dockerfile`)

- Base `python:3.12-slim`.
- Instala `HU11_UI/requirements.txt` (Streamlit fijado en `1.66.1`).
- Copia `app.py` y `api_client.py`.
- Expone `8501`, define `HEALTHCHECK` contra `/_stcore/health` y arranca
  Streamlit en modo headless con límite de subida de 10 MB.

### 5.3 Orquestación (`docker-compose.yml`)

| Servicio | Imagen | Puerto | Depende de |
|---|---|---|---|
| `api` | `hu11-traffic-api:1.0.0` | `127.0.0.1:8000` | — |
| `ui` | `hu11-traffic-ui:1.0.0` | `127.0.0.1:8501` | `api` (healthy) |

- La API se publica solo en localhost; la UI la consume por la red interna en
  `http://api:8000`.
- Ambos servicios tienen `healthcheck`, `restart: unless-stopped` y límites de
  tiempo de arranque (90 s para la API por la carga de TensorFlow).

### 5.4 Comandos

```powershell
# Construir y levantar
docker compose up --build -d

# Estado y logs
docker compose ps
docker compose logs -f

# Bajar
docker compose down
```

URLs: UI <http://127.0.0.1:8501> · API <http://127.0.0.1:8000/docs>

### 5.5 Ejecución local sin Docker

```powershell
# Terminal 1 - API
cd HU10_API
uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 - UI
cd HU11_UI
streamlit run app.py
```

## 6. Criterios de aceptación

| Criterio | Evidencia |
|---|---|
| Interfaz gráfica para subir imágenes | `app.py`: pestaña *Clasificar* (archivo y cámara) |
| Consumo de la API por HTTP | `api_client.py`: `GET /health` y `POST /predict` |
| Muestra clase y confianza | `_render_result` con `st.metric`, progreso y top-5 |
| Manejo de errores de red/API | mensajes para timeout, conexión, HTTP 400/500/503 |
| Validación local de la imagen | `validate_image` + límite `MAX_UPLOAD_MB` |
| UI independiente del modelo | no importa TensorFlow; solo `requests` y Pillow |
| Contenedor de la UI | `HU11_UI/Dockerfile` (Streamlit, puerto 8501) |
| Contenedor de la API | `HU10_API/Dockerfile` (FastAPI, puerto 8000) |
| Orquestación con healthchecks | `docker-compose.yml` (`service_healthy`) |
| Pruebas automatizadas | `HU11_UI/tests/test_api_client.py` (13 tests) |

## 7. Evidencias de verificación

- Tests: `python -m pytest HU11_UI/tests tests -q` -> **27 passed**.
- Ejecución local registrada en `logs/hu11_api.out.log`:
  `GET /health` y `POST /predict?top_k=5` responden **200 OK**.
- UI local registrada en `logs/hu11_ui.out.log` en `http://127.0.0.1:8501`.
- `python -m py_compile HU10_API/main.py HU11_UI/app.py HU11_UI/api_client.py`
  sin errores.

## 8. Limitaciones y pendientes

- **Docker no estaba instalado en la máquina de desarrollo**, por lo que las
  imágenes no se construyeron ni se ejecutó `docker compose up` en ese entorno.
  El `docker-compose.yml` y los Dockerfiles están listos para validarse donde
  Docker esté disponible.
- La primera petición a `/predict` puede tardar más mientras TensorFlow
  inicializa; el `start_period` del healthcheck (90 s) lo contempla.
- Para producción: restringir CORS, fijar versiones exactas de todas las
  dependencias, usar un proxy inverso y versionar el modelo con checksum.
