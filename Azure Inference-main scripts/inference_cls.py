import requests
import json
import base64
endpoint="https://depi-r3-project-cnn.italynorth.inference.ml.azure.com/score"
key="GKxIvCh6ZsBbWZdUbucDwjGUXp0t2dmdHAHoAASeJYmwHlHHViSZJQQJ99BKAAAAAAAAAAAAINFRAZMLgBGj"

headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
with open("image.png", "rb") as f:
    image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
data={"image": image_base64}
response = requests.post(endpoint, headers=headers, data=json.dumps(data))
print("Response status code:", response.status_code)
print("Response content:", response.json())