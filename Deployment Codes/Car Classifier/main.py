import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, resnet50
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from typing import Dict
import warnings
import io
import base64
import json
warnings.filterwarnings('ignore')

import torch.serialization
try:
    from numpy.core.multiarray import scalar
    torch.serialization.add_safe_globals([scalar])
except ImportError:
    pass

class MultiTaskCarClassifier(nn.Module):
    """
    Multi-task car classification model for standalone inference.
    """
    def __init__(self, num_makes: int, num_models: int, num_years: int, backbone: str = "efficientnet_b0"):
        super(MultiTaskCarClassifier, self).__init__()
        
        self.num_makes = num_makes
        self.num_models = num_models
        self.num_years = num_years
        self.backbone_name = backbone


        if backbone == "resnet50":
            self.backbone = resnet50(pretrained=False)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            self.backbone = efficientnet_b0(pretrained=False)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")


        self.make_head = self._create_head(self.feature_dim, num_makes)
        self.model_head = self._create_head(self.feature_dim, num_models)
        self.year_head = self._create_head(self.feature_dim, num_years)

    def _create_head(self, in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, out_features)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {
            'make': self.make_head(features),
            'model': self.model_head(features),
            'year': self.year_head(features)
        }

label_mappings = {}
label_names={}
def init():
    global model, label_mappings,label_names
    
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    supported_ext = ['.pt', '.pth']
    files = os.listdir(model_dir)
    model_path = None
    
    for f in files:
        if any(f.endswith(ext) for ext in supported_ext):
            model_path = os.path.join(model_dir, f)
            break
    

    if model_path is None:
        for root, dirs, files_in_dir in os.walk(model_dir):
            for f in files_in_dir:
                if any(f.endswith(ext) for ext in supported_ext):
                    model_path = os.path.join(root, f)
                    break
            if model_path:
                break

    if model_path is None:
        raise RuntimeError("No model file found in AZUREML_MODEL_DIR.")
    
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    arch_info = checkpoint['model_architecture']

    model = MultiTaskCarClassifier(
        num_makes=arch_info['num_makes'],
        num_models=arch_info['num_models'],
        num_years=arch_info['num_years'],
        backbone=arch_info['backbone']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    label_names= checkpoint['label_names']
    for task, encoder_data in checkpoint['label_encoders'].items():
        if 'classes' in encoder_data and encoder_data['classes']:
            classes = encoder_data['classes']
            label_mappings[task] = {i: str(cls) for i, cls in enumerate(classes)}
    
    print(f"Model loaded successfully!")
    print(f"  - Makes: {arch_info['num_makes']} classes")
    print(f"  - Models: {arch_info['num_models']} classes")
    print(f"  - Years: {arch_info['num_years']} classes")
    print(f"  - Backbone: {arch_info['backbone']}")

def preprocess_image(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def run(raw_data):
    try:
        # Parse input
        data = json.loads(raw_data) if isinstance(raw_data, (str, bytes, bytearray)) else raw_data
        
        image_base64 = data.get("image") or data.get("image_base64")
        if not image_base64:
            return {"error": "Missing 'image' or 'image_base64' key in request JSON."}
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except Exception:
            return {"error": "Invalid base64 image payload."}
        
        img_tensor = preprocess_image(image_bytes)

        with torch.no_grad():
            outputs = model(img_tensor)
        print("got outputs")
        make_predt = outputs['make'].argmax(dim=1).cpu().item()
        model_predt = outputs['model'].argmax(dim=1).cpu().item()
        year_pred = outputs['year'].argmax(dim=1).cpu().item()

        make_pred = int(label_mappings['make'].get(make_predt, f"Unknown (ID: {make_predt})"))
        model_pred = int(label_mappings['model'].get(model_predt, f"Unknown (ID: {model_predt})"))
        make_name=label_names['make_names'].get(make_pred, f"Unknown (ID: {make_pred})")
        model_name=label_names['model_names'].get(model_pred, f"Unknown (ID: {model_pred})")

        make_confidence = torch.nn.functional.softmax(outputs['make'], dim=1)[0][make_predt].item()
        model_confidence = torch.nn.functional.softmax(outputs['model'], dim=1)[0][model_predt].item()
        year_confidence = torch.nn.functional.softmax(outputs['year'], dim=1)[0][year_pred].item()

        return {
            "predictions": {
                "make": {
                    "class_id": make_pred,
                    "class_name": make_name,
                    "confidence": make_confidence
                },
                "model": {
                    "class_id": model_pred,
                    "class_name": model_name,
                    "confidence": model_confidence
                },
                "year": {
                    "class_id": year_pred,
                    "class_name": year_pred,
                    "confidence": year_confidence
                }
            }
        }
    
    except Exception as e:
        return {"error": str(e)}