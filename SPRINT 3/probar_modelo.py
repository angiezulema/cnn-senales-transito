import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import tensorflow as tf

HU07_ROOT = Path(__file__).resolve().parent
if str(HU07_ROOT) not in sys.path:
    sys.path.insert(0, str(HU07_ROOT))

# IMPORTANTE: Importar la loss custom ANTES de cargar el modelo
from src.model import load_trained_model

def main():
    parser = argparse.ArgumentParser(description="Probar modelo GTSRB de HU07")
    parser.add_argument("--imagen", required=True, help="Ruta a la imagen de la señal")
    parser.add_argument("--modelo", default="final/model_final.keras", help="Ruta al modelo (.keras o carpeta saved_model)")
    parser.add_argument("--topk", type=int, default=5, help="Mostrar top-k predicciones")
    args = parser.parse_args()

    # Cargar nombres de clases
    with open(HU07_ROOT / "final/class_names.json", encoding="utf-8") as f:
        class_names = json.load(f)

    # Cargar config de preprocesado
    with open(HU07_ROOT / "final/preprocessing.json", encoding="utf-8") as f:
        prep = json.load(f)

    h = int(prep.get("img_height", 32))
    w = int(prep.get("img_width", 32))
    scale = float(prep.get("scale", 1.0/255.0))

    print("="*50)
    print("PROBANDO MODELO HU07 - GTSRB")
    print("="*50)
    print(f"Imagen: {args.imagen}")
    print(f"Modelo: {args.modelo}")
    print(f"Preprocesado: {w}x{h} RGB, escala={scale}")
    print("="*50)

    # Cargar imagen
    img = Image.open(args.imagen).convert("RGB")
    img_resized = img.resize((w,h))
    x = np.array(img_resized).astype(np.float32)
    x = x * scale
    x = np.expand_dims(x, axis=0)  # (1, 32, 32, 3)

    # Cargar modelo CON la loss custom registrada
    modelo_path = HU07_ROOT / args.modelo
    print("Cargando modelo...")
    if modelo_path.is_dir():
        # saved_model
        model = tf.saved_model.load(str(modelo_path))
        infer = model.signatures["serving_default"]
        out = infer(tf.convert_to_tensor(x))
        preds = list(out.values())[0].numpy()
    else:
        # .keras - usar load_trained_model que registra la loss
        model = load_trained_model(str(modelo_path))
        preds = model.predict(x, verbose=0)
    
    # Top-k
    probs = preds[0]
    top_idx = probs.argsort()[-args.topk:][::-1]

    print("\nRESULTADOS:")
    print("-"*50)
    for i, idx in enumerate(top_idx):
        clase = str(idx)
        nombre = class_names.get(clase, f"Clase {clase}")
        print(f"{i+1}. Clase {clase} | {nombre} | {probs[idx]*100:.2f}%")
    print("-"*50)
    print(f"PREDICCIÓN PRINCIPAL: Clase {top_idx[0]} - {class_names.get(str(top_idx[0]), '')} ({probs[top_idx[0]]*100:.2f}%)")

if __name__ == "__main__":
    main()