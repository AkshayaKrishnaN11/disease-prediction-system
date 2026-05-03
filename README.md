# 🏥 Disease Prediction & Medical Diagnosis System

An ML-powered system for predicting diseases from patient data and chest X-rays.

## Features

- **5 Tabular Disease Predictions** — Diabetes, Heart Disease, Kidney Disease, Liver Disease, Breast Cancer (XGBoost + Random Forest)
- **Chest X-Ray Pneumonia Detection** — ResNet-50 CNN with transfer learning
- **Explainability** — SHAP feature contributions (tabular) + Grad-CAM heatmaps (X-ray)
- **Interactive Web App** — Streamlit dashboard with prediction forms, training controls, and metrics viewer

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python -m data.download_datasets

# 3. Train models
python -m train.train_tabular --disease all
python -m train.train_cnn --epochs 15

# 4. Run the app
streamlit run app.py
```

## Project Structure

```
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── data/
│   ├── download_datasets.py    # Dataset downloader
│   └── raw/                    # Raw CSV datasets
├── src/
│   ├── config.py               # Global configuration
│   ├── preprocessing/
│   │   ├── tabular.py          # Tabular data preprocessing
│   │   └── image.py            # X-ray image preprocessing
│   ├── models/
│   │   └── cnn_trainer_utils.py  # CNN model utilities
│   ├── explainability/
│   │   ├── shap_explainer.py   # SHAP explanations
│   │   └── gradcam.py          # Grad-CAM heatmaps
│   └── utils/
│       └── metrics.py          # Evaluation metrics & plots
├── train/
│   ├── train_tabular.py        # Tabular model trainer
│   └── train_cnn.py            # CNN trainer
└── saved_models/               # Trained model artifacts
```

## Tech Stack

| Component        | Technology                    |
|------------------|-------------------------------|
| Tabular ML       | XGBoost, Random Forest        |
| Deep Learning    | PyTorch, ResNet-50            |
| Explainability   | SHAP, Grad-CAM                |
| Frontend         | Streamlit                     |
| Data Processing  | Pandas, Scikit-learn, OpenCV  |
