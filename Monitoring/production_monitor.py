# Monitoring/production_monitor.py
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from alerts import AlertSystem

class ProductionModelMonitor:
    def __init__(self):
        self.client = MlflowClient()
        self.alert_system = AlertSystem()
        self.prediction_data = []
        self.baseline_metrics = {
            "classification_accuracy": 0.82,  # Your model's test accuracy
            "classification_loss": 0.45,      # Your model's test loss
            "response_time_threshold": 2.0    # seconds
        }
    
    def log_prediction(self, model_name, input_data, prediction, actual=None, latency=0):
        """
        Log production predictions for monitoring
        """
        timestamp = datetime.now()
        
        prediction_record = {
            "timestamp": timestamp,
            "model_name": model_name,
            "input_shape": str(input_data.shape) if hasattr(input_data, 'shape') else "unknown",
            "prediction": prediction,
            "actual": actual,
            "latency": latency,
            "confidence": float(np.max(prediction)) if isinstance(prediction, (list, np.ndarray)) else float(prediction),
            "predicted_class": int(np.argmax(prediction)) if isinstance(prediction, (list, np.ndarray)) else int(prediction)
        }
        
        self.prediction_data.append(prediction_record)
        
        # Check for anomalies
        self._check_anomalies(prediction_record)
        
        # Log batch to MLflow every 20 predictions
        if len(self.prediction_data) >= 20:
            self._log_monitoring_batch()
    
    def _check_anomalies(self, prediction_record):
        """Check for data drift and performance issues"""
        
        # Check latency
        if prediction_record["latency"] > self.baseline_metrics["response_time_threshold"]:
            self.alert_system.send_alert(
                f"High latency detected: {prediction_record['latency']:.2f}s for {prediction_record['model_name']}"
            )
        
        # Check confidence (if available)
        if "confidence" in prediction_record and prediction_record["confidence"] < 0.5:
            self.alert_system.send_alert(
                f"Low confidence prediction: {prediction_record['confidence']:.3f}"
            )
    
    def _log_monitoring_batch(self):
        """Log a batch of monitoring data to MLflow"""
        if self.prediction_data:
            df = pd.DataFrame(self.prediction_data)
            
            # Calculate monitoring metrics
            metrics = {
                "monitoring_batch_size": len(df),
                "avg_confidence": df['confidence'].mean(),
                "avg_latency": df['latency'].mean(),
                "max_latency": df['latency'].max(),
                "predictions_per_minute": len(df) / 5  # Assuming 5-minute batches
            }
            
            # Calculate accuracy if actual values are available
            if df['actual'].notna().any():
                correct_predictions = (df['predicted_class'] == df['actual']).sum()
                metrics["monitoring_accuracy"] = correct_predictions / len(df)
                
                # Check for performance degradation
                if "monitoring_accuracy" in metrics:
                    accuracy_drop = self.baseline_metrics["classification_accuracy"] - metrics["monitoring_accuracy"]
                    if accuracy_drop > 0.10:  # 10% drop threshold
                        self.alert_system.send_alert(
                            f"Performance degradation detected: Accuracy dropped by {accuracy_drop:.3f}"
                        )
            
            # Log to MLflow as a monitoring run
            with mlflow.start_run(run_name=f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_metrics(metrics)
                mlflow.log_params({
                    "monitoring_timestamp": datetime.now().isoformat(),
                    "data_points": len(df),
                    "monitoring_type": "production_predictions"
                })
                
                # Save the monitoring data as artifact
                monitoring_file = f"monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                df.to_csv(monitoring_file, index=False)
                mlflow.log_artifact(monitoring_file)
            
            print(f"Logged monitoring batch: {metrics}")
            self.prediction_data = []  # Clear the batch
    
    def calculate_data_drift(self, current_data, reference_data):
        """
        Calculate data drift between current and reference data
        """
        try:
            from scipy import stats
            
            # Simple statistical drift detection
            drift_metrics = {}
            
            # Compare distributions (example with confidence scores)
            if 'confidence' in current_data and 'confidence' in reference_data:
                _, p_value = stats.ks_2samp(
                    reference_data['confidence'], 
                    current_data['confidence']
                )
                drift_metrics['confidence_drift_pvalue'] = p_value
                drift_metrics['confidence_drift_detected'] = p_value < 0.05
            
            # Log drift metrics
            if drift_metrics:
                with mlflow.start_run(run_name=f"drift_detection_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                    mlflow.log_metrics(drift_metrics)
                    mlflow.log_param("drift_check_timestamp", datetime.now().isoformat())
                
                if drift_metrics.get('confidence_drift_detected', False):
                    self.alert_system.send_alert("Data drift detected in prediction confidence!")
            
            return drift_metrics
            
        except Exception as e:
            print(f"Drift detection error: {e}")
            return {}

# Initialize the monitor
production_monitor = ProductionModelMonitor()