# Sprint 3 — Reconocimiento de Señales de Tráfico (GTSRB)

Sistema Inteligente de Reconocimiento de Señales de Tráfico para la Asistencia
Vial de Conductores. El sprint integra el modelo entrenado (HU07), la API de
inferencia (HU10) y la interfaz Streamlit desplegable con Docker (HU11).

```
HU07 (modelo CNN) -> HU10 (API FastAPI) -> HU11 (UI Streamlit + Docker)
```

## Estado del Sprint 3

| Componente | Estado | Evidencia |
|---|---|---|
| Modelo CNN (HU07) | Completado | `final/model_final.keras`, accuracy test 98.39 % |
| API FastAPI (HU10) | Completado y probado en local | `HU10_API/main.py`, `HU10_API/HU10_Informe.md` |
| Interfaz Streamlit (HU11) | Completada y probada en local | `HU11_UI/app.py`, `HU11_UI/HU11_Informe.md` |
| Docker API (HU10) | Dockerfile listo, pendiente de build | `HU10_API/Dockerfile` |
| Docker UI (HU11) | Dockerfile listo, pendiente de build | `HU11_UI/Dockerfile` |
| Orquestación | Lista, pendiente de validar con Docker | `docker-compose.yml` |

### Ejecución rápida con Docker

```powershell
docker compose up --build -d
# UI:  http://127.0.0.1:8501
# API: http://127.0.0.1:8000/docs
docker compose down
```

> **Nota:** Docker no está instalado en el equipo de desarrollo actual; las
> imágenes aún no se construyeron. Ver `HU11_UI/HU11_Informe.md` para el detalle
> de la UI, el contrato de la API y los pendientes.

### Ejecución local sin Docker

```powershell
# Terminal 1 - API (puerto 8000)
cd HU10_API
uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 - UI (puerto 8501)
cd HU11_UI
streamlit run app.py
```

Pruebas: `python -m pytest HU11_UI/tests tests -q` -> 27 passed.

---

# HU07 — Ajuste de hiperparámetros y entrenamiento completo

Carpeta autocontenida para reproducir la búsqueda de hiperparámetros, el
entrenamiento final y la exportación del modelo CNN de señales de tráfico
(GTSRB, 43 clases). Incluye todo lo necesario para que otra persona continúe
con el sprint de despliegue (HU08).

> **Estado:** ver `HANDOFF.md` para el avance actual y los artefactos generados.

## Contenido

```
HU07/
├── README.md                  # esta guía
├── HANDOFF.md                 # estado, decisiones y siguientes pasos
├── requirements.txt           # dependencias (Python 3.10-3.13)
├── setup/
│   ├── setup_windows.ps1      # crea .venv e instala dependencias
│   └── check_env.py           # valida Python, TF, datos y hardware
├── config/
│   ├── base.yaml              # configuración base (réplica HU06)
│   └── search_space.yaml      # espacio de búsqueda
├── src/                       # código independiente (no importa ../src)
│   ├── config.py              # configuración, rutas y semillas
│   ├── model.py               # CNN configurable + loss con label smoothing
│   ├── data_loader.py         # CSV, resolución de rutas, caché uint8, tf.data
│   ├── train_experiment.py    # entrena 1 configuración
│   ├── run_search.py          # random search reanudable
│   ├── final_train.py         # entrenamiento final con la mejor config
│   ├── evaluate_final.py      # evaluación sobre test.csv
│   ├── export_model.py        # SavedModel + TFLite + contratos
│   └── report.py              # tabla comparativa y gráficos
├── data_manifests/            # copia de los CSV + hashes (sin imágenes)
├── experiments/               # un directorio por experimento
├── results/                   # search_results.csv, best_config.json, gráficos
├── final/                     # modelo final, métricas de test y exportaciones
├── docs/
│   ├── HU07_Informe_Tecnico.md
│   ├── HU07_resultados.md     # generado por report.py
│   └── deployment_readiness.md
├── logs/                      # logs de ejecución en segundo plano
└── tests/                     # pruebas del pipeline HU07
```

## Requisitos

- **Python 3.10 – 3.13** (recomendado 3.13). **Python 3.14 no es compatible con TensorFlow.**
- 16 GB de RAM recomendados (el caché de imágenes ocupa ~200 MB).
- **GPU:** en **Windows nativo TensorFlow es exclusivamente CPU** en esta
  versión (2.21). Tampoco se usa la iGPU AMD Radeon (ROCm es solo Linux y no
  cubre iGPUs). El soporte NVIDIA/CUDA en Windows se interrumpió tras TF 2.10;
  para GPU NVIDIA habría que usar WSL2 con `tensorflow[and-cuda]`. El código no
  requiere cambios: entrenaría en GPU si el entorno la expone.
- ~1 GB libre en disco para caché y artefactos.

## Instalación (Windows)

```powershell
# Desde la raíz del proyecto
powershell -ExecutionPolicy Bypass -File HU07\setup\setup_windows.ps1

# Si no tienes Python 3.13:
powershell -ExecutionPolicy Bypass -File HU07\setup\setup_windows.ps1 -InstallPython

# Activar entorno
cd HU07
.\.venv\Scripts\Activate.ps1
```

## Datos

HU07 usa los CSV del proyecto (`../data/*.csv`) y resuelve las imágenes en
varias raíces. Los CSV originales referencian `GTSRB/...`; el resolver prueba:

1. `<raíz>/GTSRB/...`
2. `<raíz>/archive/...` (formato del dataset descargado)
3. `<raíz>/...`
4. `data/aumentadas/...` para las imágenes aumentadas

Por eso hay que indicar la carpeta que contiene `archive/`:

```powershell
python setup\check_env.py --data-root "C:\Users\angie\Downloads\HU03"
```

También se puede fijar la variable de entorno:

```powershell
$env:HU07_DATA_ROOT = "C:\Users\angie\Downloads\HU03"
```

O editando `data.extra_roots` en `config/base.yaml`.

## Ejecución completa

```powershell
# 0. Verificar entorno y datos
python setup\check_env.py --data-root "C:\Users\angie\Downloads\HU03"

# 1. Random search (8 configuraciones x 8 épocas, batch 128, subconjunto 12k/4k)
#    ≈ 45 min en CPU (Ryzen 5 5500H). Es reanudable.
python src\run_search.py --trials 8 --epochs 8 `
    --limit-train 12000 --limit-val 4000 `
    --override batch_size=128 `
    --data-root "C:\Users\angie\Downloads\HU03"

# 2. Entrenamiento final (mejor config, dataset completo, hasta 60 épocas)
python src\final_train.py --epochs 60 --patience 10 `
    --data-root "C:\Users\angie\Downloads\HU03"

# 3. Evaluación sobre test.csv
python src\evaluate_final.py --data-root "C:\Users\angie\Downloads\HU03"

# 4. Exportación para despliegue
python src\export_model.py --data-root "C:\Users\angie\Downloads\HU03" --int8

# 5. Reporte comparativo
python src\report.py
```

### Ejecución en segundo plano (larga duración)

```powershell
New-Item -ItemType Directory -Force logs | Out-Null
# Si el venv se creo en la ruta corta, usar ese python (ver .venv_path.txt):
$py = "C:\Users\angie\AppData\Local\hu07-venv\Scripts\python.exe"
Start-Process -FilePath $py -ArgumentList @(
  "src\run_search.py", "--trials", "8", "--epochs", "8",
  "--limit-train", "12000", "--limit-val", "4000",
  "--override", "batch_size=128",
  "--data-root", "C:\Users\angie\Downloads\HU03"
) -RedirectStandardOutput "logs\search.out.log" `
  -RedirectStandardError "logs\search.err.log" -NoNewWindow
```

El search es **reanudable**: si se interrumpe, al relanzarlo omite los
experimentos ya completados (misma configuración).

## Scripts

| Script | Función |
|---|---|
| `setup/check_env.py` | Diagnóstico de Python, TF, RAM, GPU y cobertura de datos |
| `src/train_experiment.py` | Entrena una configuración puntual |
| `src/run_search.py` | Random search reanudable + `results/best_config.json` |
| `src/final_train.py` | Entrenamiento final con la mejor configuración |
| `src/evaluate_final.py` | Métricas y matriz de confusión sobre test |
| `src/export_model.py` | SavedModel, TFLite float32/dynamic-range/int8 |
| `src/report.py` | `results/search_results.csv` y gráficos comparativos |

### Opciones útiles

```powershell
# Un solo experimento con overrides
python src\train_experiment.py --exp-id prueba --epochs 3 `
    --override learning_rate=0.0005 --override batch_size=32 `
    --data-root "C:\Users\angie\Downloads\HU03"

# Prueba rápida con subconjuntos (smoke test)
python src\run_search.py --trials 2 --epochs 1 `
    --limit-train 500 --limit-val 200 --no-save-models `
    --data-root "C:\Users\angie\Downloads\HU03"

# Búsqueda con menos presupuesto
python src\run_search.py --trials 5 --epochs 6 `
    --limit-train 8000 --limit-val 3000 --override batch_size=128 `
    --data-root "C:\Users\angie\Downloads\HU03"
```

## Tests

```powershell
python -m pytest tests -v
```

Cubren configuración, overrides, resolución de rutas, caché, carga de datos,
arquitectura, loss con label smoothing y generación de trials.

## Reproducibilidad

- Semilla global configurable (`seed` en `base.yaml`, por defecto 42).
- Cada experimento guarda `config.json`, `history.json`, `metrics.json`,
  `log.csv` y `curves.png`.
- Los trials se generan con `random.Random(seed)`, por lo que el conjunto de
  configuraciones es reproducible.
- `deterministic: true` activa `tf.config.experimental.enable_op_determinism()`
  (más lento en CPU).

## Solución de problemas

| Error | Causa / solución |
|---|---|
| `No matching distribution found for tensorflow` | Python 3.14+: usa 3.13 y recrea `.venv`. |
| `No se cargó ninguna imagen` | `--data-root` no apunta a la carpeta con `archive/`. |
| `GPU AMD no detectada` | Esperado en Windows nativo; TF 2.21 corre en CPU. |
| El caché no se actualiza | Cambió el CSV o las rutas; el nombre del `.npy` incluye un hash, se regenera solo. |
| Entrenamiento muy lento | Reduce `--trials`/`--epochs` o usa `--limit-train`. Para acelerar con GPU NVIDIA sería necesario WSL2 (`tensorflow[and-cuda]`), no Windows nativo. |
| `ModuleNotFoundError: src` | Ejecuta los scripts desde `HU07/` (o usa `python -m`). |

## Privacidad y datos

El dataset GTSRB no se versiona. Solo se copian los CSV (manifiestos) y hashes
a `data_manifests/`; las imágenes permanecen en su ubicación original.
