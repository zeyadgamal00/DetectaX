import base64
import json
import requests
from PIL import Image
import io
import base64
import json
import os
from pathlib import Path

# ===Azure Details HERE===
ENDPOINT_URL = "ENDPOINT HERE"
API_KEY = "YOUR KEY"

def _encode_image(path: str) -> str:
    """Encode image to base64 string."""
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def run_inference(image_path: str, output_image_path: str = "inference_output.png"):
    """Run inference on an image and save the annotated output."""
    payload = {"image_base64": _encode_image(image_path)}
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # Get Response from Endpoint
    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    result = response.json()

    if "error" in result:
        raise RuntimeError(result["error"])
    
    # Save annotated image
    annotated = base64.b64decode(result["image_base64"])
    Path(output_image_path).write_bytes(annotated)

    return result["predictions"]


if __name__ == "__main__":
    predictions = run_inference("image.png")
    print(json.dumps(predictions, indent=2))