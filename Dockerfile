# ============================================
# Backend Dockerfile — FastAPI + ML Models
# Multi-stage build with CPU-only PyTorch
# ============================================
FROM python:3.12-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 1: Install dependencies ──
FROM base AS deps

COPY requirements.txt .

# Install CPU-only PyTorch (much smaller image) then the rest
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Final runtime image ──
FROM deps AS runtime

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY saved_models/ ./saved_models/
COPY .env.example ./.env

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run FastAPI with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
