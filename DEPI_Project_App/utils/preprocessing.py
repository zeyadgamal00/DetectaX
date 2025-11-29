"""
Image preprocessing utilities for DEPI system
"""

import numpy as np
from PIL import Image


def preprocess_image(image, target_size=(32, 32)):
    """
    Preprocess image for model inference

    Args:
        image: PIL Image object
        target_size: Target size for resizing (tuple)
                     (*** Updated default size to 32x32 ***)
    Returns:
        numpy array: Preprocessed image
    """

    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image_array = np.array(image)
    image_array = image_array.astype(np.float32) / 255.0

    return image_array


def normalize_image(image_array, mean=None, std=None):
    """
    Normalize image with mean and std
    (Not needed if model was trained on [0, 1] data)
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    normalized = (image_array - mean) / std
    return normalized


def prepare_for_classification(image, model_type='default'):
    """
    Prepare image for classification model
    (*** Updated target_size to 32x32 ***)
    """

    processed = preprocess_image(image, target_size=(32, 32))
    processed = np.expand_dims(processed, axis=0)

    return processed


def prepare_for_detection(image, model_type='default'):
    """
    Prepare image for object detection model
    (This remains unchanged for now)
    """
    processed = preprocess_image(image, target_size=(416, 416))

    if model_type == 'yolo':
        processed = processed[..., ::-1]         # Convert RGB → BGR if needed

    processed = np.expand_dims(processed, axis=0)

    return processed


def validate_image(image):
    """
    Validate image for processing
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")

    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError("Image dimensions cannot be zero")

    return True
