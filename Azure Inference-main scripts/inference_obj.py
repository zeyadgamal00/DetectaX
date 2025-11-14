import base64
import json
import requests
from PIL import Image
import io
import base64
import json
import os
from pathlib import Path

ENDPOINT_URL = "https://depi-r3-object-recognition.germanywestcentral.inference.ml.azure.com/score"
API_KEY = "FzPOfaS5PylbpI1uOtqzmZEA9WwYWzBfSk1Xd9oclk8CGZYdPPTVJQQJ99BKAAAAAAAAAAAAINFRAZML1AyP"

def _encode_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def run_inference(image_path: str, output_image_path: str = "inference_output.png"):
    payload = {"image_base64": _encode_image(image_path)}
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    result = response.json()

    if "error" in result:
        raise RuntimeError(result["error"])

    annotated = base64.b64decode(result["image_base64"])
    Path(output_image_path).write_bytes(annotated)

    return result["predictions"]


if __name__ == "__main__":
    predictions = run_inference("image.png")
    print(json.dumps(predictions, indent=2))