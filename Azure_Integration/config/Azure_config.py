"""
Azure ML Endpoint Configuration
NEVER commit this file to version control!
"""

# Classification Model Endpoint
CLASSIFICATION_ENDPOINT = "https://your-classification-endpoint.azureml.net/score"
CLASSIFICATION_API_KEY = "your-classification-api-key-here"

# Object Detection Model Endpoint  
DETECTION_ENDPOINT = "https://your-detection-endpoint.azureml.net/score"
DETECTION_API_KEY = "your-detection-api-key-here"

# Monitoring Settings
MONITORING_ENABLED = True
ALERTING_ENABLED = True