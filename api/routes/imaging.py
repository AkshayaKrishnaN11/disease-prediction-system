"""
POST /predict/xray — Chest X-ray pneumonia prediction endpoint.
Accepts an uploaded image, returns prediction + Grad-CAM heatmap.
"""

import io
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from api.schemas import XRayPredictionResponse, ErrorResponse
from api.dependencies import load_cnn_model, get_device
from src.preprocessing.image import get_val_transforms
from src.explainability.gradcam import generate_gradcam_base64

router = APIRouter(prefix="/predict", tags=["X-Ray Prediction"])


@router.post(
    "/xray",
    response_model=XRayPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Predict pneumonia from chest X-ray",
    description="Upload a chest X-ray image to get a pneumonia prediction with Grad-CAM heatmap overlay.",
)
async def predict_xray(file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG)")):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Upload JPEG or PNG.",
        )

    # Read image bytes
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Decode image
    nparr = np.frombuffer(contents, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if original_image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Load model
    try:
        model = load_cnn_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    device = get_device()

    # Preprocess
    transform = get_val_transforms()
    augmented = transform(image=original_image)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        prob_pneumonia = float(probs[0][1])
        confidence = float(probs[0].max())

    label = "Pneumonia" if prediction == 1 else "Normal"

    # Grad-CAM
    gradcam_b64 = None
    try:
        gradcam_b64 = generate_gradcam_base64(
            model, input_tensor, original_image,
            target_layer="layer4",
            target_class=prediction,
        )
    except Exception as e:
        print(f"  ⚠ Grad-CAM failed: {e}")

    return XRayPredictionResponse(
        prediction=prediction,
        prediction_label=label,
        probability=round(prob_pneumonia, 4),
        confidence=round(confidence, 4),
        gradcam_heatmap=gradcam_b64,
    )
