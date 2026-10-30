# HU07 — Informe técnico: ajuste de hiperparámetros y entrenamiento completo

## 1. Resumen

Este informe documenta la búsqueda de hiperparámetros y el entrenamiento final
de la CNN de clasificación de señales de tráfico (GTSRB, 43 clases). El punto de
partida es el modelo de HU06, que alcanzó `val_accuracy = 0.9995` tras 34 épocas.

Objetivos:

1. Explorar sistemáticamente el espacio de hiperparámetros con un presupuesto
   acotado de cómputo (CPU).
2. Seleccionar la mejor configuración con un criterio reproducible.
3. Reentrenar el modelo final y evaluarlo una única vez sobre `test.csv`.
4. Exportar artefactos listos para el sprint de despliegue (HU08).

## 2. Hardware y restricciones

| Componente | Detalle |
|---|---|
| CPU | AMD Ryzen 5 5500H, 4 núcleos / 8 hilos |
| RAM | 16 GB |
| GPU | AMD Radeon integrada (8 GB compartidos, no utilizable) |
| Framework | TensorFlow 2.21 — **CPU-only en Windows nativo** |

En **Windows nativo TensorFlow 2.21 es exclusivamente CPU**: no usa la iGPU
AMD (ROCm es solo Linux y no cubre Radeon integradas) y tampoco GPU NVIDIA
(el soporte CUDA nativo en Windows se interrumpió tras TF 2.10; requeriría
WSL2 con `tensorflow[and-cuda]`). El presupuesto se dimensionó para CPU:
búsqueda corta (8 épocas por trial, subconjunto 12k/4k, batch 128) y
refinamiento antes del entrenamiento final. Benchmark: ~40 s/época en el
subconjunto y ~2 min/época en el dataset completo. XLA (`jit_compile`) resultó
más lento y se deja desactivado.

## 3. Metodología

### 3.1 Datos

| Partición | Filas CSV | Uso |
|---|---|---|
| `train_aumentado.csv` | 38 616 | Entrenamiento (originales + aumentadas) |
| `val.csv` | 7 842 | Validación y selección de hiperparámetros |
| `test.csv` | 12 630 | Solo evaluación final |

El conjunto de prueba **no** se usa durante la búsqueda ni el entrenamiento.
Las imágenes se redimensionan a 32×32 RGB y se normalizan a [0, 1]. Se cachean
como `uint8` para evitar reprocesar en cada trial.

### 3.2 Espacio de búsqueda

| Hiperparámetro | Valores |
|---|---|
| `learning_rate` | 3e-4, 5e-4, 1e-3 |
| `batch_size` | 32, 64, 128 |
| `dropout_conv` | 0.20, 0.25, 0.30 |
| `dropout_dense` | 0.40, 0.50, 0.60 |
| `dense_units` | 256, 512 |
| `optimizer` | Adam, AdamW |
| `weight_decay` (AdamW) | 1e-4 |
| `label_smoothing` | 0.0, 0.05 |

Arquitectura convolucional fija (32→64→128 filtros) para comparar configuraciones
sobre una base estable.

### 3.3 Estrategia

1. **Baseline (E00):** configuración de HU06 para referencia.
2. **Random search:** 8 configuraciones muestreadas con semilla 42, 8 épocas
   sobre subconjunto (12 000 train / 4 000 val) con batch 128, early stopping
   con paciencia 3.
3. **Selección:** mayor `val_accuracy`; desempate por `val_f1_macro` y luego
   menor tiempo de entrenamiento.
4. **Entrenamiento final:** mejor configuración sobre el **dataset completo**,
   hasta 60 épocas, early stopping con paciencia 10, `ReduceLROnPlateau` y
   checkpoint del mejor modelo.
5. **Evaluación:** una pasada sobre `test.csv` con accuracy, precision, recall,
   F1 macro/weighted, reporte por clase y matriz de confusión.

### 3.4 Callbacks

| Callback | Configuración |
|---|---|
| EarlyStopping | `val_accuracy`, paciencia 3 (búsqueda) / 10 (final), restaura mejores pesos |
| ReduceLROnPlateau | `val_loss`, factor 0.5, paciencia 3, `min_lr=1e-6` |
| ModelCheckpoint | Mejor `val_accuracy` |
| CSVLogger | Un `log.csv` por experimento |
| TerminateOnNaN | Detiene ante pérdidas no finitas |

### 3.5 Reproducibilidad

Semilla global 42 (`random`, `numpy`, `tf.keras.utils.set_random_seed`). Cada
experimento guarda configuración, historial, métricas y curvas. El muestreo de
trials usa `random.Random(seed)`.

## 4. Resultados

Los resultados se generan automáticamente en `docs/HU07_resultados.md`
(tabla ordenada) y quedan plasmados en `results/` y `final/`.

### 4.1 Comparación de experimentos

Ver `docs/HU07_resultados.md` y `results/comparison_val_accuracy.png`.

### 4.2 Configuración seleccionada

Ver `results/best_config.json` (incluye `_meta` con el experimento de origen y
las métricas de selección).

### 4.3 Entrenamiento final

Ver `final/history.json` y `final/curves.png`.

### 4.4 Evaluación sobre test

Ver `final/test_metrics.json`, `final/classification_report.txt` y
`final/confusion_matrix.png`.

### 4.5 Comparación con HU06

| Modelo | Val accuracy | Épocas | Test accuracy |
|---|---|---|---|
| HU06 (referencia) | 0.9995 | 34 | — |
| HU07 (final) | ver `final/metrics.json` | ver `final/metrics.json` | ver `final/test_metrics.json` |

## 5. Discusión

- La validación de GTSRB a 32×32 está cerca del techo (~99.9%), por lo que las
  diferencias entre configuraciones serán pequeñas. El criterio secundario
  (F1 macro y costo) gana relevancia.
- Un `val_accuracy` muy alto puede ocultar clases difíciles; el F1 macro y la
  matriz de confusión permiten detectar ese caso.
- El entrenamiento en CPU limita el número de trials; en GPU se recomienda
  ampliar el espacio (más filtros, mixup/cutout, cosine decay).

## 6. Amenazas a la validez

- **Sobreajuste a la validación:** se mitiga eligiendo una sola configuración y
  evaluando una única vez en test.
- **Presupuesto de búsqueda:** 8 trials no cubren todo el espacio; los
  resultados son indicativos, no óptimos globales.
- **Hardware:** los tiempos de CPU no son comparables con GPU.
- **Datos aumentados:** la partición aumentada local contiene 7 249 imágenes; el
  resto de originales se resuelve desde una carpeta externa, lo que puede variar
  entre entornos.

## 7. Conclusiones

> Completar al finalizar la ejecución.

1. **Mejor configuración (búsqueda):** `E01_rnd` — LR 0.001, batch 32,
   dropout_conv 0.20/dropout_dense 0.60, dense_units 512, Adam (sin label
   smoothing). Ver `results/best_config.json`.
2. **Entrenamiento final (completo):** mejor `val_accuracy = 0.9990`
   (época 16/26, early stopping en 26), `val_f1_macro = 0.9994`,
   tiempo 291 min en CPU.
3. **Evaluación test:** `accuracy = 0.9839`, `precision_macro = 0.9755`,
   `recall_macro = 0.9738`, `f1_macro = 0.9743`, `f1_weighted = 0.9838`
   (12 630 muestras). Ver `final/test_metrics.json`.
4. **Exportación:** SavedModel, TFLite float32/dynamic-range/int8. Verificación
   sobre 300 muestras aleatorias (semilla 42): los 3 artefactos TFLite igualan
   la exactitud de Keras (0.9767) con MAE de probabilidades < 1.5e-4.
   Ver `final/export_report.json` y `docs/HU07_resultados.md`.
5. Los artefactos para despliegue están en `final/` y el contrato en
   `docs/deployment_readiness.md`.

## 8. Reproducción

```powershell
python setup\check_env.py --data-root "C:\Users\angie\Downloads\HU03"
python src\run_search.py --trials 8 --epochs 8 `
    --limit-train 12000 --limit-val 4000 --override batch_size=128 `
    --data-root "C:\Users\angie\Downloads\HU03"
python src\final_train.py --epochs 60 --patience 10 `
    --data-root "C:\Users\angie\Downloads\HU03"
python src\evaluate_final.py --data-root "C:\Users\angie\Downloads\HU03"
python src\export_model.py --data-root "C:\Users\angie\Downloads\HU03" --int8
python src\report.py
```
