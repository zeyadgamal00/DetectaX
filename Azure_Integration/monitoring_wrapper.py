# azure_integration/monitoring_wrapper.py
import sys
import os

sys.path.append('../Monitoring')
sys.path.append('..')

from Monitoring.production_monitor import production_monitor, ProductionModelMonitor
from Monitoring.alerts import AlertSystem
from classification_client import classification_client
from detection_client import detection_client

class AzureMLMonitor:
    def __init__(self):
        self.classification_client = classification_client
        self.detection_client = detection_client
        self.monitor = production_monitor
    
    def get_system_status(self):
        """Get overall system status"""
        alert_summary = self.monitor.alert_system.get_alert_summary(hours=24)
        
        status = {
            "classification_endpoint": "Configured" if hasattr(self.classification_client, 'endpoint_url') else "Not configured",
            "detection_endpoint": "Configured" if hasattr(self.detection_client, 'endpoint_url') else "Not configured",
            "monitoring_active": True,
            "recent_alerts": alert_summary,
            "total_predictions": len(self.monitor.prediction_data)
        }
        
        return status
    
    def test_connectivity(self):
        """Test connectivity to Azure endpoints"""
        results = {}
        
        # Test classification endpoint (simple connectivity test)
        try:
            # This would be a proper test - for now just check configuration
            if hasattr(self.classification_client, 'endpoint_url'):
                results['classification'] = "Configured"
            else:
                results['classification'] = "Not configured"
        except Exception as e:
            results['classification'] = f"Error: {e}"
        
        # Test detection endpoint
        try:
            if hasattr(self.detection_client, 'endpoint_url'):
                results['detection'] = "Configured"
            else:
                results['detection'] = "Not configured"
        except Exception as e:
            results['detection'] = f"Error: {e}"
        
        return results

# Create global instance
azure_monitor = AzureMLMonitor()