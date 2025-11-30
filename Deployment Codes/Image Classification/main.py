import json
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import io
import base64
import keras
def init():
    global model    
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    files = os.listdir(model_dir)

    supported_ext = [".keras", ".h5", ".pb"]

    model_path = None

    for f in files:
        if any(f.endswith(ext) for ext in supported_ext):
            model_path = os.path.join(model_dir, f)
            break
        
    model=keras.models.load_model(model_path)
    if model_path is None:
        raise RuntimeError("No model file found in AZUREML_MODEL_DIR.")
    print(f"Loading model from: {model_path}")



def preprocess_image(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def run(raw_data):
    try:
        data = json.loads(raw_data)
        image_base64 = data.get("image")
        if not image_base64:
            return json.dumps({"error": "Missing 'image' key in request JSON."})

        image_bytes = base64.b64decode(image_base64)
        img_array = preprocess_image(image_bytes)

        preds = model.predict(img_array)
        predicted_class = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))

        return{
            "predicted_class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        return json.dumps({"error": str(e)})
