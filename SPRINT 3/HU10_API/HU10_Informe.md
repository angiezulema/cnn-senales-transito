# HU10 - Implementación de API con FastAPI para Reconocimiento de Señales

## 1. Descripción

La HU10 implementa una API REST con FastAPI que sirve el modelo de Red Neuronal
Convolucional (CNN) entrenado y exportado en HU07 para clasificar 43 señales de
tráfico del dataset GTSRB.

La API recibe una imagen, la valida, la preprocesa con el mismo contrato usado
durante el entrenamiento y devuelve la clase predicha junto con su nivel de
confianza. El código de la API es independiente del código de entrenamiento y
consume únicamente los artefactos finales de `HU07` en modo lectura:

| Artefacto | Ruta | Uso en la API |
|---|---|---|
| Modelo Keras | `final/model_final.keras` | Inferencia (`tf.keras.models.load_model`) |
| Nombres de clase | `final/class_names.json` | Traducir `class_id` a nombre legible |
| Preprocesamiento | `final/preprocessing.json` | Tamaño, escala y modo de color |
| Configuración base | `config/base.yaml` | Verificación de `32x32`, 3 canales y 43 clases |

El contrato de entrada extraído es: imagen **RGB**, redimensionada a **32×32**
con interpolación **bilineal**, convertida a `float32` y normalizada dividiendo
entre 255 (`scale = 0.00392156862745098`), con layout **NHWC** `[1, 32, 32, 3]`.

## 2. Arquitectura

La aplicación se estructura en un único módulo `main.py`:

- **Configuración**: instancia de `FastAPI` con título, descripción y versión.
- **CORS**: `CORSMiddleware` habilitado para permitir el consumo desde un
  frontend.
- **Carga de artefactos**: en el evento `startup` se leen `class_names.json` y
  `preprocessing.json`, y se carga el modelo con
  `tf.keras.models.load_model(..., compile=False)`. Se usa `compile=False`
  porque el modelo fue compilado con la loss personalizada
  `SparseCategoricalCrossentropyLS`, innecesaria para inferencia.
- **Estado del modelo**: una variable global `model` actúa como indicador de
  disponibilidad y es consultada por ambos endpoints.
- **Manejo de errores**: `HTTPException` con códigos 400, 500 y 503 según el
  caso.

### 2.1 Endpoint `GET /health`

Verifica que el servicio esté activo y si el modelo se cargó correctamente.

Respuesta:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

Si el modelo no pudo cargarse, `model_loaded` es `false` y `/predict` responde
con HTTP 503.

### 2.2 Endpoint `POST /predict`

Recibe un archivo de imagen mediante `multipart/form-data` en el campo `file`.
Flujo interno:

1. Verifica que el modelo esté cargado (si no, HTTP 503).
2. Lee los bytes del archivo y rechaza archivos vacíos (HTTP 400).
3. Valida y decodifica la imagen con `PIL.Image`. Si está corrupta o no es una
   imagen, responde HTTP 400.
4. Convierte a RGB, redimensiona a 32×32 con interpolación bilineal, convierte a
   `float32`, multiplica por `1/255` y añade la dimensión de batch.
5. Ejecuta `model.predict(tensor, verbose=0)`.
6. Calcula `argmax` sobre las 43 probabilidades softmax y devuelve el resultado.
7. Si ocurre un error interno durante la inferencia, responde HTTP 500.

## 3. Criterios de Aceptación Cumplidos

| Criterio | Evidencia |
|---|---|
| La API recibe una imagen por HTTP | `POST /predict` con `UploadFile` (`multipart/form-data`) |
| Devuelve la clase predicha | Campo `class_id` (entero) y `class_name` (string) |
| Devuelve el nivel de confianza | Campo `confidence`, redondeado a 4 decimales |
| Preprocesamiento idéntico al entrenamiento | RGB, 32×32 bilineal, `float32`, escala 1/255, NHWC |
| Manejo de imágenes inválidas | Validación con PIL y HTTP 400 para archivos corruptos o no imagen |
| Manejo de modelo no disponible | HTTP 503 si `model` es `None` |
| Manejo de errores internos | HTTP 500 durante la inferencia |
| Estado del servicio | `GET /health` con `status` y `model_loaded` |
| Integración con frontend | `CORSMiddleware` habilitado |
| Artefactos de HU07 intactos | Solo lectura de `final/` y `config/base.yaml` |

## 4. Instrucciones de Ejecución

Desde la carpeta `HU10_API`:

```powershell
# 1. Crear y activar un entorno virtual (opcional pero recomendado)
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Levantar el servidor
uvicorn main:app --reload
```

La API estará disponible en `http://127.0.0.1:8000`. La documentación
interactiva se genera automáticamente en:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

> El modelo se carga una sola vez al iniciar el servidor. La primera petición
> puede tardar más mientras TensorFlow inicializa sus recursos.

## 5. Contrato de API

### 5.1 `POST /predict`

Solicitud:

```http
POST /predict HTTP/1.1
Content-Type: multipart/form-data; boundary=...
```

Campo requerido: `file` (imagen JPG, PNG u otro formato soportado por Pillow).

Ejemplo con `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@senal_stop.jpg"
```

Respuesta exitosa (HTTP 200):

```json
{
  "class_id": 14,
  "class_name": "Stop",
  "confidence": 0.9987
}
```

Respuestas de error:

```json
{
  "detail": "El archivo no es una imagen válida."
}
```

```json
{
  "detail": "El modelo no está disponible. Consulte /health."
}
```

```json
{
  "detail": "Error interno durante la predicción."
}
```

### 5.2 `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true
}
```

## 6. Consideraciones Técnicas

- El modelo se carga con `compile=False` para evitar depender de la loss
  personalizada definida en el proyecto de entrenamiento. La inferencia solo
  requiere la arquitectura y los pesos.
- Las rutas se resuelven de forma relativa a `main.py` (`HU10_API/main.py` →
  `../final/...`), por lo que la API funciona sin rutas absolutas de la máquina
  original.
- El tamaño de entrada y la escala se leen en tiempo de ejecución desde
  `final/preprocessing.json`, manteniendo un único contrato entre entrenamiento
  y servicio. Si el archivo faltara, se usan los valores de `config/base.yaml`
  como respaldo.
- La API no escribe, mueve ni elimina archivos de `HU07`; todos los accesos son
  de lectura.
- Para producción se recomienda restringir `allow_origins` del middleware CORS,
  fijar versiones exactas de dependencias y ejecutar Uvicorn detrás de un proxy
  inverso.
