import requests
import json
import base64
import io
import os
from PIL import Image

try:
    from utils.helpers import CLASS_NAMES
except ImportError:
    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

CNN_ENDPOINT = ""
CNN_KEY = ""

OD_ENDPOINT = ""
OD_KEY = ""

def _pil_to_base64(image: Image.Image, format="PNG") -> str:
    try:
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        raise

def classify_image(image: Image.Image, use_api=False):
    print("Sending request to Azure CNN endpoint...")

    headers = {
        "Authorization": f"Bearer {CNN_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "image": _pil_to_base64(image)
    }

    try:
        response = requests.post(CNN_ENDPOINT, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        if "error" in result:
            raise Exception(result["error"])

        predicted_index = int(result["predicted_class"])
        predicted_class_name = CLASS_NAMES[predicted_index]
        confidence = float(result["confidence"])

        return {
            "class": predicted_class_name,
            "confidence": confidence,
            "method": "azure_cnn"
        }

    except requests.exceptions.RequestException as e:
        print(f"Azure CNN request failed: {e}")
        return {"class": "Connection Error", "confidence": 0.0, "method": "error"}
    except Exception as e:
        print(f"Error processing CNN response: {e}")
        return {"class": "Prediction Error", "confidence": 0.0, "method": "error"}

def detect_objects(image: Image.Image, threshold=0.5, use_api=False):
    print(f"Sending request to Azure OD endpoint with threshold {threshold}...")

    headers = {
        "Authorization": f"Bearer {OD_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "image_base64": _pil_to_base64(image),
        "conf": float(threshold)
    }

    try:
        response = requests.post(OD_ENDPOINT, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()

        if "error" in result:
            raise Exception(result["error"])

        raw_predictions = result.get("predictions", [])
        formatted_detections = []

        for pred in raw_predictions:
            box = pred['box']
            formatted_detections.append({
                "label": pred['name'],
                "confidence": float(pred['confidence']),
                "bbox": [
                    int(box['x1']),
                    int(box['y1']),
                    int(box['x2']),
                    int(box['y2'])
                ]
            })

        return formatted_detections

    except requests.exceptions.RequestException as e:
        print(f"Azure OD request failed: {e}")
        return []
    except Exception as e:
        print(f"Error processing OD response: {e}")
        return []
