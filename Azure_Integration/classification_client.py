# azure_integration/classification_client.py
import requests
import json
import base64
import time
import os
import sys

# Add parent directory to path to import monitoring
sys.path.append('../Monitoring')
sys.path.append('..')

from Monitoring.production_monitor import production_monitor
from config.Azure_config import CLASSIFICATION_ENDPOINT, CLASSIFICATION_API_KEY

class ClassificationClient:
    def __init__(self, endpoint_url=None, api_key=None):
        self.endpoint_url = endpoint_url or CLASSIFICATION_ENDPOINT
        self.api_key = api_key or CLASSIFICATION_API_KEY
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def predict(self, image_path=None, image_base64=None):
        """
        Make classification prediction with monitoring
        
        Args:
            image_path: Path to image file
            image_base64: Base64 encoded image string
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
            
            # Prepare payload (matches your main_cls.py format)
            data = {"image": image_base64}
            
            # Make prediction request
            response = requests.post(
                self.endpoint_url, 
                headers=self.headers, 
                data=json.dumps(data),
                timeout=30  # 30 second timeout
            )
            response.raise_for_status()
            
            latency = time.time() - start_time
            result = response.json()
            
            # Extract prediction data
            predicted_class = result.get("predicted_class")
            confidence = result.get("confidence", 0)
            
            # Log to monitoring system
            production_monitor.log_prediction(
                model_name="azure-classification-model",
                input_data=image_base64[:100] + "..." if len(image_base64) > 100 else image_base64,  # Truncate for logging
                prediction=predicted_class,
                confidence=confidence,
                latency=latency,
                actual=None  # Can be provided if you have ground truth
            )
            
            print(f"Prediction successful: Class {predicted_class} with {confidence:.3f} confidence")
            return result
            
        except requests.exceptions.RequestException as e:
            latency = time.time() - start_time
            error_msg = f"Request failed: {str(e)}"
            production_monitor.alert_system.send_alert(error_msg, level="ERROR")
            raise Exception(error_msg)
        
        except Exception as e:
            latency = time.time() - start_time
            error_msg = f"Prediction error: {str(e)}"
            production_monitor.alert_system.send_alert(error_msg, level="ERROR")
            raise

    def batch_predict(self, image_paths):
        """
        Make batch predictions
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path=image_path)
                results.append(result)
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")
                results.append({"error": str(e), "image_path": image_path})
        
        return results

# Create global instance
classification_client = ClassificationClient()