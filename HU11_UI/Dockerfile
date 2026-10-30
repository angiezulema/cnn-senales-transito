# HU11 - Imagen de la interfaz Streamlit.
# Construir desde la raíz del proyecto:
#   docker build -f HU11_UI/Dockerfile -t hu11-traffic-ui:1.0.0 .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY HU11_UI/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY HU11_UI/app.py HU11_UI/api_client.py ./

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.maxUploadSize=10", \
     "--browser.gatherUsageStats=false"]
