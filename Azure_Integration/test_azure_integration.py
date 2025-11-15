# azure_integration/test_azure_integration.py
import sys
import os

# Add current directory to path
sys.path.append('.')

from Azure_Integration.monitoring_wrapper import azure_monitor

def test_azure_integration():
    """Test the Azure integration setup"""
    print("Testing Azure Integration...")
    
    # Check system status
    status = azure_monitor.get_system_status()
    print("System Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test connectivity
    connectivity = azure_monitor.test_connectivity()
    print("\nEndpoint Connectivity:")
    for service, result in connectivity.items():
        print(f"  {service}: {result}")
    
    print("\nAzure integration test completed!")
    print("Next: Add your actual endpoint URLs and API keys to azure_config.py")

if __name__ == "__main__":
    test_azure_integration()