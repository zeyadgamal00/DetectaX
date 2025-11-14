# Import all necessary modules from MLflow files
import mlflow
from tracking_setup import setup_mlflow_tracking, start_experiment_run
from experiment_logging import log_training_metrics
from model_registry import ModelRegistry

def test_mlflow_integration():
    """
    Test the MLflow setup with Azure ML
    """
    print("Testing MLflow Integration")
    
    # Setup tracking
    ws = setup_mlflow_tracking()
    
    # Test classification experiment
    with start_experiment_run("test-classification-experiment", "test_run"):
        log_training_metrics({
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.82,
            "f1_score": 0.825
        })
        
        mlflow.log_param("model_type", "test_classification")
        mlflow.log_param("dataset", "cifar10")
        
        print("Classification experiment logging test completed")
    
    # Test detection experiment  
    with start_experiment_run("test-detection-experiment", "test_run"):
        log_training_metrics({
            "mAP50": 0.72,
            "mAP50-95": 0.45,
            "precision": 0.68,
            "recall": 0.65
        })
        
        mlflow.log_param("model_type", "test_detection")
        mlflow.log_param("dataset", "coco")
        
        print("Detection experiment logging test completed")
    
    # Test model registry
    if ws:
        registry = ModelRegistry(ws)
        models = registry.list_model_versions("image-classification-model")
        print("Model registry test completed")
    
    print("All MLflow tests completed successfully!")

if __name__ == "__main__":
    test_mlflow_integration()