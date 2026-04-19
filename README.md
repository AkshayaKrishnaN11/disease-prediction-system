# 🏥 Disease Prediction & Medical Diagnosis System

An end-to-end ML-powered system for predicting diseases from patient data and chest X-rays, with a modern Next.js dashboard and full deployment pipeline.

## 🤖 Models

### Tabular (XGBoost + Random Forest)
| Disease | Dataset | Features |
|---------|---------|----------|
| Diabetes | Pima Indians | Glucose, BMI, Age, etc. |
| Heart Disease | Cleveland | Chest pain, Cholesterol, etc. |
| Kidney Disease | UCI CKD | Albumin, Blood Pressure, etc. |
| Liver Disease | ILPD | Bilirubin, Proteins, etc. |
| Breast Cancer | Wisconsin | Radius, Texture, Perimeter, etc. |

### Computer Vision (ResNet-50)
| Task | Dataset | Architecture |
|------|---------|-------------|
| Pneumonia Detection | Chest X-Ray | ResNet-50 (Transfer Learning) |

## 📁 Project Structure

```
disease_anti/
├── src/                  # Core ML code (config, preprocessing, explainability)
├── train/                # Training scripts (tabular + CNN)
├── api/                  # FastAPI backend (predict, metrics, health)
├── frontend/             # Next.js 14 dashboard (TypeScript + Tailwind)
├── tests/                # Pytest test suite (API + model tests)
├── saved_models/         # Trained model artifacts + metrics
├── data/                 # Dataset downloader + raw data
├── Dockerfile            # Backend Docker image
├── docker-compose.yml    # Full-stack Docker compose
├── railway.toml          # Railway deployment config
├── .github/workflows/    # GitHub Actions CI pipeline
└── requirements.txt      # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Train Models
```bash
python -m train.train_tabular --disease all
python -m train.train_cnn --epochs 15
```

### 3. Start Backend API
```bash
python -m api.main
```
API docs: http://localhost:8000/docs

### 4. Start Frontend
```bash
cd frontend
npm install
npm run dev
```
Dashboard: http://localhost:3000

### 5. Run Tests
```bash
pytest tests/ -v
```

## 🐳 Docker

```bash
docker-compose up --build
```
Backend: http://localhost:8000 · Frontend: http://localhost:3000

## ☁️ Deployment

| Component | Platform | Config |
|-----------|----------|--------|
| Backend API | Railway | `railway.toml` |
| Frontend | Vercel | `frontend/vercel.json` |
| CI/CD | GitHub Actions | `.github/workflows/ci.yml` |

## 📊 Explainability
- **SHAP** — Feature importance for tabular models
- **Grad-CAM** — Heatmap overlays on X-ray predictions

## 🛠 Tech Stack
- **ML**: XGBoost, Random Forest, PyTorch (ResNet-50)
- **Backend**: FastAPI + Pydantic + Uvicorn
- **Frontend**: Next.js 14 + TypeScript + Tailwind + Recharts
- **Tracking**: MLflow
- **Testing**: Pytest + FastAPI TestClient
- **DevOps**: Docker, GitHub Actions, Railway, Vercel
