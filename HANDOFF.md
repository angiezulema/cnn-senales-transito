# HU07 — Handoff

Documento para que otra persona (o el siguiente sprint) continúe el trabajo sin
depender del contexto original.

## 1. Objetivo de HU07

Ajustar hiperparámetros de la CNN de señales de tráfico, ejecutar el
entrenamiento completo con la mejor configuración y dejar el modelo exportado
para el sprint de despliegue (HU08).

## 2. Estado actual

> Actualizar esta sección al cerrar cada fase.

| Fase | Estado | Evidencia |
|---|---|---|
| Estructura autocontenida HU07 | Completada | esta carpeta |
| Entorno Python + dependencias | Completado | `.venv_path.txt` |
| Verificación de entorno/datos | Completado | `setup/check_env.py` |
| Random search | Completado | `experiments/`, `results/search_results.csv` |
| Entrenamiento final | Completado | `final/model_final.keras` |
| Evaluación en test | Completado | `final/test_metrics.json` (98.39 % accuracy) |
| Exportación | Completado | `final/saved_model/`, `final/*.tflite` |
| Informe y resultados | Completado | `docs/HU07_Informe_Tecnico.md`, `docs/HU07_resultados.md` |
| API de inferencia (HU10) | Completado y probado en local | `HU10_API/main.py`, `HU10_API/HU10_Informe.md` |
| Interfaz Streamlit (HU11) | Completada y probada en local | `HU11_UI/app.py`, `HU11_UI/HU11_Informe.md` |
| Docker / orquestación | Dockerfiles y `docker-compose.yml` listos; **pendiente de build** | `HU10_API/Dockerfile`, `HU11_UI/Dockerfile`, `docker-compose.yml` |

### Estado del Sprint 3 (HU10 + HU11)

- API FastAPI con `/health` y `/predict` (top-k), validada localmente (logs con
  respuestas 200).
- UI Streamlit con pestañas *Clasificar*, *Información* y *Estado de la API*.
- `docker-compose.yml` orquesta `api` (8000) y `ui` (8501) con healthchecks.
- Tests: `python -m pytest HU11_UI/tests tests -q` -> **27 passed**.
- **Pendiente:** instalar Docker y validar `docker compose up --build -d`.
- Detalle completo: `HU11_UI/HU11_Informe.md`.

## 3. Decisiones tomadas

| Decisión | Elección | Motivo |
|---|---|---|
| Método de búsqueda | Random search | Espacio pequeño; reproducible sin dependencias extra |
| Presupuesto | ~5 h CPU | Equipo con GPU AMD integrada (no usable por TF en Windows) |
| Independencia | HU07 no importa `../src` | Handoff seguro y sin acoplamiento |
| Datos | CSV del proyecto + raíces configurables | No duplicar imágenes (~1 GB) |
| Despliegue | Solo artefactos + documentación | HU08 implementa API/Docker |
| Selección | `val_accuracy` > `f1_macro` > tiempo | Prioriza exactitud de validación |

## 4. Hardware de referencia

- CPU: AMD Ryzen 5 5500H (4 núcleos / 8 hilos)
- RAM: 16 GB
- GPU: AMD Radeon integrada (8 GB compartidos) — **no soportada por TensorFlow
  en Windows**; entrenamiento en CPU
- Presupuesto observado en HU07: ~2 min/época con el dataset completo
  (38 616 train / 7 842 val, batch 128, `intra_threads=4`)

> **Nota sobre GPU NVIDIA:** esta versión de TensorFlow (2.21) en **Windows
> nativo es exclusivamente CPU**. El soporte NVIDIA/CUDA en Windows se
> interrumpió tras TF 2.10 y **no** se activa por instalación de CUDA; solo es
> posible vía WSL2 con `tensorflow[and-cuda]`. No asumir aceleración por GPU
> en este entorno.

## 5. Datos

- CSV: `../data/train_aumentado.csv` (38 616 filas), `val.csv` (7 842),
  `test.csv` (12 630).
- Imágenes aumentadas: `../data/aumentadas/`.
- Imágenes originales GTSRB: **no están dentro del proyecto**. Se localizaron en
  `C:\Users\angie\Downloads\HU03\archive` (39 209 imágenes Train, 12 631 Test).
- El resolver prueba `GTSRB/...`, `archive/...` y la raíz directa; por eso basta
  `--data-root "C:\Users\angie\Downloads\HU03"`.

Si se mueve el dataset, actualizar `HU07_DATA_ROOT` o `data.extra_roots` en
`config/base.yaml`.

## 6. Cómo continuar

```powershell
cd HU07
powershell -ExecutionPolicy Bypass -File setup\setup_windows.ps1
.\.venv\Scripts\Activate.ps1

python setup\check_env.py --data-root "C:\Users\angie\Downloads\HU03"
python -m pytest tests -v

# Reanudar búsqueda (omite experimentos ya completados): ~45 min CPU
python src\run_search.py --trials 8 --epochs 8 `
    --limit-train 12000 --limit-val 4000 --override batch_size=128 `
    --data-root "C:\Users\angie\Downloads\HU03"

python src\final_train.py --epochs 60 --patience 10 `
    --data-root "C:\Users\angie\Downloads\HU03"
python src\evaluate_final.py --data-root "C:\Users\angie\Downloads\HU03"
python src\export_model.py --data-root "C:\Users\angie\Downloads\HU03" --int8
python src\report.py
```

## 7. Inventario de artefactos

| Artefacto | Ruta | Descripción |
|---|---|---|
| Config base | `config/base.yaml` | Punto de partida (réplica HU06) |
| Espacio de búsqueda | `config/search_space.yaml` | Valores explorados |
| Resultados de búsqueda | `results/search_results.csv` | Tabla ordenada por val accuracy |
| Mejor configuración | `results/best_config.json` | Entrada de `final_train.py` |
| Comparativa | `results/comparison_val_accuracy.png` | Gráfico de experimentos |
| Modelo final | `final/model_final.keras` | Modelo Keras completo |
| Métricas test | `final/test_metrics.json` | Accuracy, precision, recall, F1 |
| Reporte por clase | `final/classification_report.txt` | Detalle de las 43 clases |
| Matriz de confusión | `final/confusion_matrix.png` | Errores entre clases |
| SavedModel | `final/saved_model/` | Para TF Serving |
| TFLite | `final/model*.tflite` | Para dispositivos/edge |
| Contrato de entrada | `final/preprocessing.json` | 32×32 RGB, [0,1], NHWC |
| Nombres de clase | `final/class_names.json` | id → nombre |
| Informe | `docs/HU07_Informe_Tecnico.md` | Metodología y resultados |
| Resultados | `docs/HU07_resultados.md` | Tabla autogenerada |
| Despliegue | `docs/deployment_readiness.md` | Insumos para HU08 |

## 8. Limitaciones conocidas

- Evaluación y búsqueda usan el mismo `val.csv`; el test solo se toca al final.
- El dataset aumentado local cubre 7 249 imágenes; las 31 367 originales se
  resuelven desde la carpeta externa `HU03/archive`.
- La GPU AMD no acelera TensorFlow en Windows; no intentar `tensorflow-directml`
  (TF 2.10, sin mantenimiento) sin justificarlo.
- El espacio de búsqueda es deliberadamente pequeño por presupuesto de CPU.

## 9. Siguientes pasos (Sprint 3)

1. ~~Implementar API (FastAPI) con el contrato de `docs/deployment_readiness.md`.~~ Completado (HU10).
2. ~~Contenerizar con Docker (CPU) usando `requirements-inference.txt`.~~ Dockerfiles listos; falta construir y validar.
3. Validar latencia/throughput con el TFLite o SavedModel exportado.
4. Definir monitoreo, versionado del modelo y política de actualización.
5. Pruebas de integración con imágenes reales y casos borde (baja luz, blur).

### Tareas pendientes concretas

- Instalar Docker Desktop / habilitar Docker Engine en la máquina de despliegue.
- Ejecutar `docker compose up --build -d` y verificar `docker compose ps`
  (ambos servicios `healthy`).
- Probar la predicción end-to-end desde la UI con una imagen real.
- Medir latencia P95 y throughput de `/predict` en el contenedor.
- Opcional: eliminar la copia obsoleta `SPRINT 3/` (duplicado sin HU11) y
  actualizar `docs/deployment_readiness.md` con la evidencia final.
