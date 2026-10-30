# Insumos para el sprint de despliegue (HU08)

Este documento define el contrato y los artefactos que HU07 deja listos para
desplegar el modelo. **La implementación de la API/Docker corresponde a HU08.**

## 1. Artefactos exportados

| Artefacto | Ruta | Uso recomendado |
|---|---|---|
| Keras | `final/model_final.keras` | Reentrenamiento, evaluación, referencia |
| SavedModel | `final/saved_model/` | TF Serving / servicio batch |
| TFLite float32 | `final/model.tflite` | Edge/CPU, máxima fidelidad |
| TFLite dynamic range | `final/model_dynamic_range.tflite` | Menor tamaño, I/O float |
| TFLite int8 | `final/model_int8.tflite` (si `--int8`) | Mínimo tamaño, validar precisión |
| Nombres de clase | `final/class_names.json` | Respuesta legible |
| Preprocesamiento | `final/preprocessing.json` | Contrato de entrada |
| Reporte de exportación | `final/export_report.json` | Tamaños y verificación |

## 2. Contrato de entrada

- Formato: imagen RGB (JPG/PNG) o tensor.
- Redimensionado: **32×32** con interpolación **bilineal**.
- Conversión: RGB (descartar alfa), `float32`.
- Normalización: dividir entre 255 → rango **[0, 1]**.
- Layout del tensor: **NHWC** `[1, 32, 32, 3]`.

```json
{
  "color_mode": "RGB",
  "img_height": 32,
  "img_width": 32,
  "resize_method": "bilinear",
  "scale": 0.00392156862745098,
  "input_layout": "NHWC",
  "input_dtype": "float32",
  "output": "softmax probabilities (43 classes)"
}
```

## 3. Contrato de salida

- Vector de 43 probabilidades (softmax, suma 1).
- Clase predicha: `argmax`.
- Respuesta sugerida:

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

## 4. Referencia de rendimiento (a completar tras evaluar)

| Métrica | Valor |
|---|---|
| Accuracy en test | ver `final/test_metrics.json` |
| F1 macro | ver `final/test_metrics.json` |
| Tamaño `.keras` | ver `final/export_report.json` |
| Tamaño TFLite float32 | ver `final/export_report.json` |
| Latencia CPU (batch 1) | medir en HU08 |
| Throughput CPU | medir en HU08 |

> El hardware de referencia (Ryzen 5 5500H, sin GPU utilizable por TF) no es
> representativo de un servidor. Medir latencia en el entorno objetivo.

## 5. Opciones de despliegue evaluadas

| Opción | Ventajas | Consideraciones |
|---|---|---|
| **FastAPI + Docker (CPU)** | Simple, portable, fácil de monitorear | Recomendada para HU08 |
| TF Serving | Gestión de modelos/versiones | Más infraestructura |
| ONNX Runtime | Buen rendimiento CPU | Requiere conversión adicional |
| TFLite en dispositivo | Bajo consumo | Precisión int8 a validar |

### Esqueleto sugerido para HU08 (no implementado)

```python
# app/main.py (referencia)
from fastapi import FastAPI, UploadFile
from PIL import Image
import io, json, numpy as np, tensorflow as tf

app = FastAPI(title="Traffic Sign Classifier")
model = tf.keras.models.load_model("final/model_final.keras")
classes = json.load(open("final/class_names.json"))

def preprocess(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((32, 32), Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0)[None, ...]

@app.post("/predict")
async def predict(file: UploadFile):
    proba = model.predict(preprocess(await file.read()), verbose=0)[0]
    top = np.argsort(proba)[::-1][:5]
    return {
        "class_id": int(top[0]),
        "class_name": classes[str(int(top[0]))],
        "confidence": float(proba[top[0]]),
        "top_k": [
            {"class_id": int(i), "class_name": classes[str(int(i))],
             "probability": float(proba[i])}
            for i in top
        ],
    }
```

```dockerfile
# Dockerfile (referencia)
FROM python:3.13-slim
WORKDIR /app
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt
COPY final/ ./final/
COPY app/ ./app/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 6. Checklist de aceptación para HU08

- [ ] API responde con el contrato de salida definido.
- [ ] Preprocesamiento idéntico al de entrenamiento (32×32, RGB, [0,1]).
- [ ] Latencia P95 medida y documentada.
- [ ] Imagen Docker reproducible y con versión fijada.
- [ ] Pruebas con imágenes reales y casos borde.
- [ ] Modelo versionado y con checksum.
- [ ] Logs y métricas básicas de servicio.
- [ ] Documento de operación (cómo desplegar, actualizar y revertir).

## 7. Advertencias

- No usar `tensorflow-directml` en producción: está limitado a TF 2.10 y sin
  mantenimiento.
- La GPU AMD integrada no acelera TF en Windows; el contenedor debe asumir CPU.
- Si se usa TFLite int8, comparar accuracy contra `final/model.tflite` antes de
  desplegar.
