# Monitoring/alerts.py
import logging
import smtplib
from email.mime.text import MimeText
import mlflow
from datetime import datetime, timedelta

class AlertSystem:
    def __init__(self):
        self.alert_thresholds = {
            "accuracy_drop": 0.10,      # 10% drop
            "high_latency": 2.0,        # seconds
            "low_confidence": 0.5,      # confidence threshold
            "data_drift_pvalue": 0.05   # statistical significance
        }
        self.alert_history = []
    
    def send_alert(self, message, level="WARNING"):
        """Send alert with different severity levels"""
        timestamp = datetime.now()
        alert_record = {
            "timestamp": timestamp,
            "message": message,
            "level": level
        }
        self.alert_history.append(alert_record)
        
        # Log to console and file
        log_message = f"{level} ALERT [{timestamp}]: {message}"
        
        if level == "CRITICAL":
            logging.critical(log_message)
        elif level == "ERROR":
            logging.error(log_message)
        else:
            logging.warning(log_message)
        
        print(log_message)
        
        # TODO: Integrate with your preferred alerting system:
        # - Email notifications
        # - Slack webhooks  
        # - Azure Monitor alerts
        # - PagerDuty
        
        self._log_alert_to_mlflow(alert_record)
    
    def _log_alert_to_mlflow(self, alert_record):
        """Log alerts to MLflow for tracking"""
        try:
            with mlflow.start_run(run_name=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}", nested=True):
                mlflow.log_params({
                    "alert_timestamp": alert_record["timestamp"].isoformat(),
                    "alert_level": alert_record["level"],
                    "alert_message": alert_record["message"]
                })
                mlflow.log_metric("alert_triggered", 1)
        except Exception as e:
            print(f"Failed to log alert to MLflow: {e}")
    
    def get_alert_summary(self, hours=24):
        """Get summary of recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alert_history if a["timestamp"] > cutoff_time]
        
        summary = {
            "total_alerts": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a["level"] == "CRITICAL"]),
            "warning_alerts": len([a for a in recent_alerts if a["level"] == "WARNING"]),
            "latest_alert": recent_alerts[-1]["message"] if recent_alerts else "No recent alerts"
        }
        
        return summary

# Initialize alert system
alert_system = AlertSystem()