# MLflow/tracking_setup.py
import mlflow
import mlflow.keras
from azureml.core import Workspace
import os
from datetime import datetime

def setup_mlflow_tracking():
    """
    Set up MLflow tracking with Azure ML workspace
    """
    try:
        # Connect to Azure ML workspace
        ws = Workspace.from_config()
        mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
        print("MLflow tracking URI set to Azure ML workspace")
        return ws
    except Exception as e:
        print(f"Failed to connect to Azure ML: {e}")
        # Fallback to local tracking
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        print("Fallback to local MLflow tracking")
        return None

def start_experiment_run(experiment_name, run_name=None):
    """
    Start a new MLflow experiment run
    """
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    print(f"Started MLflow run: {run_name} in experiment: {experiment_name}")
    return run

def log_classification_training_params(model_params, training_params):
    """
    Log parameters for classification model training
    """
    mlflow.log_params({
        **model_params,
        **training_params,
        "model_type": "classification",
        "timestamp": datetime.now().isoformat()
    })

def log_detection_training_params(model_params, training_params):
    """
    Log parameters for object detection model training
    """
    mlflow.log_params({
        **model_params,
        **training_params, 
        "model_type": "object_detection",
        "timestamp": datetime.now().isoformat()
    })