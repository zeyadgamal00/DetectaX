"""
Visualization utilities for DEPI system
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

try:
    from utils.helpers import get_class_color, CLASS_COLORS
except ImportError:
    CLASS_COLORS = {'default': '#0066CC'}


    def get_class_color(class_name):
        return CLASS_COLORS['default']


def draw_bounding_boxes(image, detections, font_size=14):
    """
    Draw bounding boxes and labels on image

    Args:
        image: PIL Image object
        detections: List of detection dictionaries
        font_size: Font size for labels

    Returns:
        PIL Image: Image with bounding boxes drawn
    """

    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        print("Arial.ttf not found, using default font.")
        font = ImageFont.load_default()

    for i, detection in enumerate(detections):
        label = detection['label']
        confidence = detection['confidence']
        bbox = detection['bbox']
        color_hex = get_class_color(label)
        color_rgb = tuple(int(color_hex.lstrip('#')[j:j + 2], 16) for j in (0, 2, 4))
        draw.rectangle(bbox, outline=color_rgb, width=3)
        label_text = f"{label}: {confidence:.2f}"

        try:
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except Exception:
            text_width = len(label_text) * (font_size * 0.6)
            text_height = font_size * 1.2
        label_bg_y = bbox[1] - text_height - 5
        if label_bg_y < 0:
            label_bg_y = bbox[1] + 1

        label_bg = [
            bbox[0], label_bg_y,
            bbox[0] + text_width + 10, label_bg_y + text_height + 4
        ]
        draw.rectangle(label_bg, fill=color_rgb)
        draw.text(
            (bbox[0] + 5, label_bg_y + 2),
            label_text,
            fill='white',
            font=font
        )

    return annotated_image


def create_confidence_chart(detections):
    """
    Create a confidence score chart for detections
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available")
        return None

    labels = [det['label'] for det in detections]
    confidences = [det['confidence'] for det in detections]
    colors = [get_class_color(label) for label in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, confidences, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Confidence')
    ax.set_title('Object Detection Confidence Scores')
    for i, v in enumerate(confidences):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center')

    plt.tight_layout()
    return fig
