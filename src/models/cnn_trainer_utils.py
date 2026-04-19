"""
Utility to build ResNet-50 for inference (loading saved weights).
Separated to avoid circular imports.
"""

import torch
import torch.nn as nn
from pathlib import Path


def build_resnet50_for_inference(model_path, device):
    """
    Build ResNet-50 and load saved weights for inference.

    Args:
        model_path: Path to the .pth checkpoint
        device: torch.device

    Returns:
        Loaded model in eval mode
    """
    from torchvision.models import resnet50

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 2)

    # Build model with same architecture as training
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model
