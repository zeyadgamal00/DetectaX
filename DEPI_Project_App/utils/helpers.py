"""
Helper functions and constants for DEPI system
"""
import os

PRIMARY_COLOR = "#0066CC"
SECONDARY_COLOR = "#00CCFF"
SUCCESS_COLOR = "#00CC00"
WARNING_COLOR = "#FF9900"
ERROR_COLOR = "#FF3333"

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
CLASS_COLORS = {
    'airplane': '#0066CC',
    'automobile': '#FF9D97',
    'bird': '#FF701F',
    'cat': '#FFB21A',
    'deer': '#CFD231',
    'dog': '#48F90A',
    'frog': '#00CC00',
    'horse': '#FF3838',
    'ship': '#00B4D8',
    'truck': '#FF9D97',
    'default': '#0066CC'
}


def get_class_color(class_name):
    """Get color for a specific class"""

    return CLASS_COLORS.get(class_name, CLASS_COLORS['default'])


def format_confidence(confidence):
    """Format confidence score as percentage"""
    return f"{confidence:.1%}"


def get_file_size(file_path):
    """Get file size in human readable format"""
    size_bytes = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def validate_image_file(file):
    """Validate uploaded image file"""
    allowed_types = ['image/jpeg', 'image/png', 'image/bmp']
    max_size = 10 * 1024 * 1024  # 10MB

    if file.type not in allowed_types:
        return False, "Invalid file type. Please upload JPEG, PNG, or BMP."

    if file.size > max_size:
        return False, "File too large. Please upload images under 10MB."

    return True, "Valid file"
