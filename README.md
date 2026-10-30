# Sprint 3 — Reconocimiento de Señales de Tráfico (GTSRB)

Sistema inteligente de reconocimiento de señales de tráfico para la asistencia vial de conductores. El sprint integra tres unidades de trabajo:

`mermaid
flowchart LR
    HU07 --> HU10 --> HU11
`

## Estado Actual del Sprint

| Componente | Estado | Evidencia |
|---|---|---|
| **Modelo CNN (HU07)** | Completado | inal/model_final.keras, accuracy test 98.39% |
| **API FastAPI (HU10)** | Completado y probado en local | HU10_API/main.py, HU10_API/HU10_Informe.md |
| **Interfaz Streamlit (HU11)** | Completada y probada en local | HU11_UI/app.py, HU11_UI/HU11_Informe.md |
| **Docker API (HU10)** | Dockerfile listo, pendiente de build | HU10_API/Dockerfile |
| **Docker UI (HU11)** | Dockerfile listo, pendiente de build | HU11_UI/Dockerfile |
| **Orquestación** | Lista, pendiente de validar con Docker | docker-compose.yml |

---

## Ejecución Rápida con Docker

`powershell
docker compose up --build -d
# UI:  http://127.0.0.1:8501
# API: http://127.0.0.1:8000/docs
docker compose down
`

> **Nota:** Las imágenes aún no se han construido. Ver HU11_UI/HU11_Informe.md para detalles de la UI, el contrato de la API y los pendientes.

## Ejecución Local sin Docker

### Terminal 1 - API (puerto 8000)

`powershell
cd HU10_API
uvicorn main:app --host 127.0.0.1 --port 8000
`

### Terminal 2 - UI (puerto 8501)

`powershell
cd HU11_UI
streamlit run app.py
`

### Pruebas

`powershell
python -m pytest HU11_UI/tests tests -q
# 27 passed
`
