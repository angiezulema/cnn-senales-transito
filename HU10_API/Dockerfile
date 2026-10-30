# HU11 - Imagen de la API FastAPI (HU10) para CPU.
# Construir desde la raíz del proyecto:
#   docker build -f HU10_API/Dockerfile -t hu11-traffic-api:1.0.0 .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# libgomp1 es requerido por los wheels de TensorFlow en Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY HU10_API/requirements-docker.txt ./HU10_API/requirements-docker.txt
RUN pip install --no-cache-dir -r HU10_API/requirements-docker.txt

# main.py resuelve PROJECT_ROOT como el padre de HU10_API -> /app.
COPY HU10_API/main.py ./HU10_API/main.py
COPY final/model_final.keras final/class_names.json final/preprocessing.json ./final/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "HU10_API.main:app", "--host", "0.0.0.0", "--port", "8000"]
