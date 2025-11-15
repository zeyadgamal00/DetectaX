# azure_integration/detection_client.py
import requests
import json
import base64
import time
import os
import sys

sys.path.append('../Monitoring')
sys.path.append('..')

from Monitoring.production_monitor import production_monitor
from config.Azure_config import DETECTION_ENDPOINT, DETECTION_API_KEY

class DetectionClient:
    def __init__(self, endpoint_url=None, api_key=None):
        self.endpoint_url = endpoint_url or DETECTION_ENDPOINT
        self.api_key = api_key or DETECTION_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def predict(self, image_path=None, image_base64=None, conf=0.25, iou=0.45):
        """
        Make object detection prediction with monitoring
        """
        start_time = time.time()
        
        try:
            # Prepare image data
            if image_path:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            elif not image_base64:
                raise ValueError("Either image_path or image_base64 must be provided")
            
            # Prepare payload
            data = {
                "image_base64": image_base64,
                "conf": conf,
                "iou": iou
            }
            
            # Make prediction request
            response = requests.post(
                self.endpoint_url, 
                headers=self.headers, 
                data=json.dumps(data),
                timeout=60  # Longer timeout for detection
            )
            response.raise_for_status()
            
            latency = time.time() - start_time
            result = response.json()
            
            # Extract detection data
            predictions = result.get("predictions", [])
            num_detections = len(predictions)
            
            # Log to monitoring system
            production_monitor.log_prediction(
                model_name="azure-detection-model",
                input_data=image_base64[:100] + "..." if len(image_base64) > 100 else image_base64,
                prediction=num_detections, 
                confidence=1.0, 
                latency=latency,
                actual=None
            )
            
            print(f"Detection successful: Found {num_detections} objects")
            return result
            
        except requests.exceptions.RequestException as e:
            latency = time.time() - start_time
            error_msg = f"Detection request failed: {str(e)}"
            production_monitor.alert_system.send_alert(error_msg, level="ERROR")
            raise Exception(error_msg)
        
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Detection error: {str(e)}"
            production_monitor.alert_system.send_alert(error_msg, level="ERROR")
            raise

# Create global instance
detection_client = DetectionClient()