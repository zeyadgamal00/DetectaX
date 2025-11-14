# MLflow/experiment_logging.py
import mlflow
import matplotlib.pyplot as plt
import tempfile
import os

def log_training_metrics(metrics_dict, prefix=""):
    """
    Log training metrics to MLflow
    """
    for key, value in metrics_dict.items():
        metric_name = f"{prefix}_{key}" if prefix else key
        mlflow.log_metric(metric_name, value)
    print(f"Logged {len(metrics_dict)} metrics")

def log_classification_artifacts(history, class_names, output_dir="artifacts"):
    """
    Log classification training artifacts (plots, reports)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(plot_path)
    plt.close()
    
    mlflow.log_artifact(plot_path)
    print("Logged training history plot")
    
    # Log class names
    classes_path = os.path.join(output_dir, "class_names.txt")
    with open(classes_path, 'w') as f:
        for i, class_name in enumerate(class_names):
            f.write(f"{i}: {class_name}\n")
    
    mlflow.log_artifact(classes_path)
    print("Logged class names")

def log_detection_artifacts(results, output_dir="artifacts"):
    """
    Log object detection training artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Log detection metrics
    if hasattr(results, 'results_dict'):
        metrics_path = os.path.join(output_dir, "detection_metrics.txt")
        with open(metrics_path, 'w') as f:
            for key, value in results.results_dict.items():
                f.write(f"{key}: {value}\n")
        
        mlflow.log_artifact(metrics_path)
        print("Logged detection metrics")
    
    # Log confusion matrix if available
    try:
        if hasattr(results, 'confusion_matrix'):
            cm_path = os.path.join(output_dir, "confusion_matrix.png")
            results.confusion_matrix.plot(save_dir=output_dir)
            mlflow.log_artifact(os.path.join(output_dir, "confusion_matrix.png"))
            print("Logged confusion matrix")
    except Exception as e:
        print(f"Could not log confusion matrix: {e}")