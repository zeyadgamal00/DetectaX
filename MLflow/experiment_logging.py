# MLflow/experiment_logging.py
import mlflow
import matplotlib.pyplot as plt
import tempfile
import os
import numpy as np

def log_training_metrics(metrics_dict, prefix=""):
    """
    Log training metrics to MLflow
    """
    for key, value in metrics_dict.items():
        metric_name = f"{prefix}_{key}" if prefix else key
        mlflow.log_metric(metric_name, value)
    print(f"✅ Logged {len(metrics_dict)} metrics")

def log_classification_artifacts(history, class_names, output_dir="artifacts"):
    """
    Log classification training artifacts (plots, reports)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Learning rate schedule (if available)
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.ylabel('LR')
        plt.xlabel('Epoch')
        plt.legend()
    
    plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    mlflow.log_artifact(plot_path)
    print("Logged training history plot")
    
    # Log class names
    classes_path = os.path.join(output_dir, "class_names.txt")
    with open(classes_path, 'w') as f:
        f.write("CIFAR-10 Class Names:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{i}: {class_name}\n")
    
    mlflow.log_artifact(classes_path)
    print("Logged class names")

def log_model_summary(model, output_dir="artifacts"):
    """
    Log model architecture summary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    summary_path = os.path.join(output_dir, "model_summary.txt")
    with open(summary_path, 'w') as f:
        # Redirect model summary to file
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    mlflow.log_artifact(summary_path)
    print("Logged model summary")