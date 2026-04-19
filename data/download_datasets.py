"""
Download all datasets required for the Disease Prediction System.
Uses direct public URLs (no Kaggle API needed).

Usage:
    python -m data.download_datasets
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR

# ──────────────────────────────────────────────
# Dataset URLs (public mirrors / direct links)
# ──────────────────────────────────────────────
TABULAR_DATASETS = {
    "diabetes.csv": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "columns": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
        ],
        "description": "Pima Indians Diabetes Dataset",
    },
    "heart.csv": {
        "url": "https://raw.githubusercontent.com/datasets/heart-disease/main/data/heart.csv",
        "fallback_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "description": "Cleveland Heart Disease Dataset",
    },
    "kidney_disease.csv": {
        "url": "https://raw.githubusercontent.com/datasets/kidney-disease/main/data/kidney_disease.csv",
        "fallback_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00336/Chronic_Kidney_Disease.zip",
        "description": "UCI Chronic Kidney Disease Dataset",
    },
    "liver.csv": {
        "url": "https://raw.githubusercontent.com/datasets/indian-liver-patient/main/data/indian_liver_patient.csv",
        "fallback_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
        "description": "Indian Liver Patient Dataset",
    },
    "breast_cancer.csv": {
        "url": "https://raw.githubusercontent.com/datasets/breast-cancer-wisconsin/main/data/data.csv",
        "description": "Wisconsin Breast Cancer Dataset (from sklearn fallback)",
    },
}

XRAY_DATASET = {
    "url": "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded",
    "alt_url": "https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia",
    "description": "Chest X-Ray Images (Pneumonia) from Mendeley / Kaggle",
}


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            desc=desc or dest.name,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ⚠ Failed to download from {url}: {e}")
        return False


def generate_from_sklearn(filename: str, dest: Path):
    """Fallback: generate dataset from scikit-learn built-in datasets."""
    import pandas as pd

    if filename == "breast_cancer.csv":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["diagnosis"] = data.target  # 0 = malignant, 1 = benign
        # Remap: 0→M, 1→B to match standard Wisconsin format
        df["diagnosis"] = df["diagnosis"].map({0: "M", 1: "B"})
        df.to_csv(dest, index=False)
        print(f"  ✓ Generated {filename} from sklearn ({len(df)} rows)")
        return True

    if filename == "diabetes.csv":
        # The Pima dataset has specific column names
        cols = TABULAR_DATASETS[filename]["columns"]
        url = TABULAR_DATASETS[filename]["url"]
        try:
            df = pd.read_csv(url, header=None, names=cols)
            df.to_csv(dest, index=False)
            print(f"  ✓ Downloaded {filename} ({len(df)} rows)")
            return True
        except Exception:
            return False

    return False


def download_tabular_datasets():
    """Download all 5 tabular disease datasets."""
    print("\n" + "=" * 60)
    print("📊 Downloading Tabular Datasets")
    print("=" * 60)

    for filename, info in TABULAR_DATASETS.items():
        dest = DATA_DIR / filename
        if dest.exists():
            print(f"  ✓ {filename} already exists, skipping")
            continue

        print(f"\n  📥 {info['description']} → {filename}")

        # Try primary URL
        success = download_file(info["url"], dest, desc=filename)

        # Try fallback URL
        if not success and "fallback_url" in info:
            print(f"  ↻ Trying fallback URL...")
            success = download_file(info["fallback_url"], dest, desc=filename)

        # Try sklearn generation
        if not success:
            print(f"  ↻ Trying sklearn fallback...")
            success = generate_from_sklearn(filename, dest)

        if not success:
            print(f"  ✗ Could not download {filename}. Please download manually.")

    # Verify files
    print(f"\n{'─' * 40}")
    for filename in TABULAR_DATASETS:
        dest = DATA_DIR / filename
        status = "✓" if dest.exists() else "✗"
        print(f"  {status} {filename}")


def download_xray_dataset():
    """Download chest X-ray dataset (large ~1.2GB)."""
    print("\n" + "=" * 60)
    print("🫁 Downloading Chest X-Ray Dataset")
    print("=" * 60)

    xray_dir = DATA_DIR / "chest_xray"
    if xray_dir.exists() and any(xray_dir.rglob("*.jpeg")):
        print("  ✓ Chest X-ray dataset already exists, skipping")
        return

    zip_path = DATA_DIR / "chest_xray.zip"

    print("  📥 This is a large download (~1.2 GB)...")
    print("  ℹ  If auto-download fails, manually download from:")
    print("     https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print(f"     Extract to: {xray_dir}")

    success = download_file(
        XRAY_DATASET["url"], zip_path, desc="chest_xray.zip"
    )

    if success and zip_path.exists():
        print("  📦 Extracting...")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DATA_DIR)
            zip_path.unlink()
            print("  ✓ Extraction complete")
        except Exception as e:
            print(f"  ⚠ Extraction failed: {e}")
    else:
        print("\n  ⚠ Auto-download failed. Please download manually:")
        print("    1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("    2. Click 'Download' button")
        print(f"    3. Extract the zip to: {xray_dir}")
        print("    4. Ensure folder structure is: chest_xray/train/NORMAL/ and chest_xray/train/PNEUMONIA/")

    # Create placeholder structure if download failed
    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            (xray_dir / split / cls).mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("🏥 Disease Prediction System — Dataset Downloader")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    download_tabular_datasets()
    download_xray_dataset()

    print("\n" + "=" * 60)
    print("✅ Dataset download complete!")
    print("=" * 60)
    print("\nNext step: python -m train.train_tabular --disease all")


if __name__ == "__main__":
    main()
