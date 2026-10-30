# HU07 - Resultados de la busqueda de hiperparametros

> Generado: 2026-10-23T06:13:05

Referencia HU06: val_accuracy **0.9995** (34 epocas).

| # | Experimento | Val acc | F1 macro | Val loss | Epoca | Epocas | Tiempo (min) | LR | Batch | Drop conv/dense | Dense | Optim | WD | LS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | E01_rnd | 0.9933 | 0.9916 | 0.0271 | 8 | 8 | 5.2 | 0.001 | 32 | 0.2/0.6 | 512 | adam | 0.0 | 0.0 |
| 2 | E02_rnd | 0.9818 | 0.9795 | 0.0737 | 8 | 8 | 4.9 | 0.001 | 32 | 0.3/0.6 | 256 | adamw | 0.0001 | 0.0 |
| 3 | E00_baseline | 0.9812 | 0.9791 | 0.0723 | 8 | 8 | 5.0 | 0.001 | 128 | 0.25/0.5 | 512 | adam | 0.0 | 0.0 |
| 4 | E05_rnd | 0.9747 | 0.9791 | 0.5708 | 8 | 8 | 4.5 | 0.0003 | 32 | 0.25/0.4 | 256 | adamw | 0.0001 | 0.05 |
| 5 | E07_rnd | 0.9697 | 0.9701 | 0.1005 | 8 | 8 | 4.2 | 0.001 | 64 | 0.3/0.6 | 512 | adam | 0.0 | 0.0 |
| 6 | E03_rnd | 0.9688 | 0.9712 | 0.1408 | 8 | 8 | 4.8 | 0.0003 | 32 | 0.2/0.6 | 256 | adam | 0.0 | 0.0 |
| 7 | E06_rnd | 0.9672 | 0.9695 | 0.1361 | 8 | 8 | 4.2 | 0.0005 | 128 | 0.25/0.4 | 512 | adam | 0.0 | 0.0 |
| 8 | E04_rnd | 0.9585 | 0.9656 | 0.6240 | 8 | 8 | 4.6 | 0.0005 | 128 | 0.25/0.4 | 256 | adamw | 0.0001 | 0.05 |

## Mejor configuracion

- Experimento: **E01_rnd**
- Val accuracy: **0.9933**
- F1 macro: **0.9916**
- Val loss: 0.0271
- Epoca del mejor modelo: 8 de 8

## Evaluacion sobre el conjunto de prueba

| Metrica | Valor |
|---|---|
| Accuracy | 0.9839 |
| Precision macro | 0.9755 |
| Recall macro | 0.9738 |
| F1 macro | 0.9743 |
| F1 weighted | 0.9838 |
| Muestras | 12630 |

## Verificacion de exportaciones TFLite (muestra aleatoria test)

| Artefacto | Exactitud | Coincidencia con Keras | MAE de probabilidades |
|---|---|---|---|
| Keras (float32, /255) | 0.9767 | — | — |
| `model.tflite` (float32) | 0.9767 | 1.0000 | 1.52e-09 |
| `model_dynamic_range.tflite` | 0.9767 | 1.0000 | 1.91e-05 |
| `model_int8.tflite` (representative dataset) | 0.9767 | 1.0000 | 1.45e-04 |

*Muestra: 300 imágenes aleatorias de `test.csv` (semilla 42).*  
> El conjunto de prueba no se uso durante la busqueda ni el entrenamiento; solo en esta evaluacion final.
