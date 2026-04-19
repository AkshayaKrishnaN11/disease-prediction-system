"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN explainability.
Generates heatmap overlays on chest X-ray images to show where the model is looking.
"""

import io
import base64
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """
    Grad-CAM implementation for PyTorch CNN models.

    Usage:
        gradcam = GradCAM(model, target_layer='layer4')
        heatmap = gradcam.generate(input_tensor)
        overlay = gradcam.overlay_on_image(original_image, heatmap)
    """

    def __init__(self, model, target_layer_name: str = "layer4"):
        self.model = model
        self.model.eval()

        # Register hooks on target layer
        self.gradients = None
        self.activations = None

        target_layer = dict(model.named_modules())[target_layer_name]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    @torch.no_grad()
    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an input image tensor.

        Args:
            input_tensor: Preprocessed image tensor [1, C, H, W]
            target_class: Class index to explain (None = predicted class)

        Returns:
            Heatmap as numpy array [H, W] with values in [0, 1]
        """
        self.model.eval()

        # Enable gradients temporarily
        input_tensor.requires_grad_(True)

        with torch.enable_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            # Backward pass for target class
            self.model.zero_grad()
            output[0, target_class].backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def overlay_on_image(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.

        Args:
            original_image: Original image [H, W, 3] in RGB, uint8
            heatmap: Grad-CAM heatmap [H_cam, W_cam] in [0, 1]
            alpha: Transparency of overlay

        Returns:
            Overlaid image [H, W, 3] in RGB, uint8
        """
        # Resize heatmap to match image
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert to uint8 and apply colormap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = (
            alpha * heatmap_colored.astype(np.float32) +
            (1 - alpha) * original_image.astype(np.float32)
        ).astype(np.uint8)

        return overlay


def generate_gradcam_base64(
    model, input_tensor: torch.Tensor, original_image: np.ndarray,
    target_layer: str = "layer4", target_class: int = None,
) -> str:
    """
    Generate Grad-CAM overlay and return as base64 encoded PNG.

    Args:
        model: PyTorch CNN model
        input_tensor: Preprocessed image tensor [1, C, H, W]
        original_image: Original image [H, W, 3] in RGB
        target_layer: Name of the target layer for Grad-CAM
        target_class: Class to explain (None = predicted)

    Returns:
        Base64 encoded PNG string
    """
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor, target_class)
    overlay = gradcam.overlay_on_image(original_image, heatmap)

    # Encode to base64
    _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    b64_string = base64.b64encode(buffer).decode("utf-8")

    return b64_string


def save_gradcam_plot(
    model, input_tensor: torch.Tensor, original_image: np.ndarray,
    save_path: str, target_layer: str = "layer4", target_class: int = None,
) -> str:
    """
    Generate and save a side-by-side comparison: original vs Grad-CAM overlay.

    Returns:
        Path to saved image
    """
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor, target_class)
    overlay = gradcam.overlay_on_image(original_image, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original X-Ray", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis("off")

    plt.suptitle("Grad-CAM Explainability", fontsize=14, fontweight="bold")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path
