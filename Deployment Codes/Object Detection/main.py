import ultralytics
import os
from PIL import Image
import io
import json
import base64
import cv2
import numpy as np
def init():
    global model    
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    files = os.listdir(model_dir)

    supported_ext = [".pt"]

    model_path = None
    for f in files:
        if any(f.endswith(ext) for ext in supported_ext):
            model_path = os.path.join(model_dir, f)
            break

    if model_path is None:
        raise RuntimeError("No model file found in AZUREML_MODEL_DIR.")

    print(f"Loading model from: {model_path}")

    model = ultralytics.YOLO(model_path)
    
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img

def run(raw_data):
    try:
        data = json.loads(raw_data) if isinstance(raw_data, (str, bytes, bytearray)) else raw_data

        img_b64 = data.get("image_base64")
        if not img_b64:
            return {"error": "Missing 'image_base64' field"}

        conf = float(data.get("conf", 0.25))
        iou = float(data.get("iou", 0.45))

        try:
            img_bytes = base64.b64decode(img_b64, validate=True)
        except Exception:
            return {"error": "Invalid base64 in 'image_base64'"}

        if not img_bytes:
            return {"error": "Empty image payload"}

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Failed to decode image"}

        h, w = img.shape[:2]

        results = model(img, conf=conf, iou=iou)
        result = results[0]
        annotated = result.plot()
        ok, buf = cv2.imencode(".png", annotated)
        if not ok:
            return {"error": "Failed to encode annotated image"}
        out_b64 = base64.b64encode(buf).decode("utf-8")
        preds_json = json.loads(result.to_json())

        return {
            "image_base64": out_b64,
            "image_format": "png",
            "image_shape": {"width": int(w), "height": int(h)},
            "predictions": preds_json,
            "classes": result.names
        }
    except Exception as e:
        return {"error": str(e)}