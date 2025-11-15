import requests
import json
import base64

# Add Azure endpoint and key here (Classification Model)
endpoint="ENDPOINT URL"
key="YOUR KEY"
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

# Read image file and encode it to base64
with open("image.png", "rb") as f:
    image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
data={"image": image_base64}

# Send POST request to Azure endpoint
response = requests.post(endpoint, headers=headers, data=json.dumps(data))

# Print response
print("Response status code:", response.status_code)
print("Response content:", response.json())