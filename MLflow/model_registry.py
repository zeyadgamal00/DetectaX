# MLflow/model_registry.py
import mlflow
from azureml.core import Model, Workspace
import os
import json

class ModelRegistry:
    def __init__(self, workspace):
        self.ws = workspace
        self.registered_models = {}
    
    def register_classification_model(self, model_path, run_id, metrics, description=""):
        """
        Register a new classification model version
        """
        try:
            # Register with Azure ML
            model = Model.register(
                workspace=self.ws,
                model_path=model_path,
                model_name="image-classification-model",
                tags={
                    "accuracy": f"{metrics.get('accuracy', 0):.4f}",
                    "framework": "keras",
                    "task": "classification"
                },
                description=description,
                model_framework=Model.Framework.KERAS
            )
            
            # Also log with MLflow
            with mlflow.start_run(run_id=run_id):
                mlflow.keras.log_model(mlflow.keras.load_model(model_path), "classification_model")
                mlflow.set_tag("registered_model_id", model.id)
            
            self.registered_models["classification"] = model
            print(f"Registered classification model: {model.id}")
            return model
            
        except Exception as e:
            print(f"Failed to register classification model: {e}")
            return None
    
    def register_detection_model(self, model_path, run_id, metrics, description=""):
        """
        Register a new object detection model version using custom MLflow logging
        """
        try:
            # Register with Azure ML
            model = Model.register(
                workspace=self.ws,
                model_path=model_path,
                model_name="object-detection-model", 
                tags={
                    "mAP50": f"{metrics.get('mAP50', 0):.4f}",
                    "framework": "ultralytics",
                    "task": "object_detection"
                },
                description=description
            )
            
            # Log to MLflow as a generic model with custom metadata
            with mlflow.start_run(run_id=run_id):
                # Log the model file as an artifact
                mlflow.log_artifact(model_path, "model")
                
                # Log model metadata
                mlflow.log_param("model_type", "yolo_object_detection")
                mlflow.log_param("framework", "ultralytics")
                mlflow.log_metrics(metrics)
                
                # Create a requirements.txt for the model
                requirements = [
                    "ultralytics",
                    "torch",
                    "torchvision",
                    "opencv-python",
                    "Pillow",
                    "numpy"
                ]
                
                with open("requirements.txt", "w") as f:
                    for req in requirements:
                        f.write(f"{req}\n")
                
                mlflow.log_artifact("requirements.txt")
                mlflow.set_tag("registered_model_id", model.id)
            
            self.registered_models["detection"] = model
            print(f"Registered detection model: {model.id}")
            return model
            
        except Exception as e:
            print(f"Failed to register detection model: {e}")
            return None
    
    def list_model_versions(self, model_name):
        """
        List all versions of a registered model
        """
        try:
            models = Model.list(self.ws, name=model_name)
            print(f"Versions of {model_name}:")
            for model in models:
                print(f"  - Version {model.version}: {model.id}")
            return models
        except Exception as e:
            print(f"Failed to list models: {e}")
            return []