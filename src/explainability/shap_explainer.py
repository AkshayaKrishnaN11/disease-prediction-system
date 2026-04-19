"""
SHAP explainability for tabular ML models (XGBoost, Random Forest).
Generates feature importance explanations for model predictions.
"""

import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def get_shap_explanations(model, X_test, feature_names=None, max_samples=100):
    """
    Compute SHAP values for a tree-based model.

    Args:
        model: Trained XGBoost or RandomForest model
        X_test: Test feature matrix (DataFrame or ndarray)
        feature_names: List of feature names
        max_samples: Max samples to explain (for performance)

    Returns:
        dict with:
            - shap_values: raw SHAP values array
            - feature_importance: dict of {feature: mean_abs_shap}
            - base_value: expected value (bias)
    """
    # Sample for performance
    if len(X_test) > max_samples:
        indices = np.random.choice(len(X_test), max_samples, replace=False)
        X_sample = X_test.iloc[indices] if hasattr(X_test, "iloc") else X_test[indices]
    else:
        X_sample = X_test

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle multi-output (for RF which returns list)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 (positive)

    # Feature importance = mean absolute SHAP value
    if feature_names is None:
        if hasattr(X_test, "columns"):
            feature_names = list(X_test.columns)
        else:
            feature_names = [f"Feature {i}" for i in range(X_test.shape[1])]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = {
        name: float(val) for name, val in zip(feature_names, mean_abs_shap)
    }

    # Sort by importance
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "shap_values": shap_values,
        "feature_importance": feature_importance,
        "base_value": float(explainer.expected_value) if np.isscalar(explainer.expected_value)
                       else float(explainer.expected_value[1]),
        "X_sample": X_sample,
    }


def explain_single_prediction(model, X_single, feature_names=None):
    """
    Generate SHAP explanation for a single prediction (used by API).

    Args:
        model: Trained model
        X_single: Single sample (1D array or 1-row DataFrame)

    Returns:
        dict with feature contributions for this prediction
    """
    import pandas as pd

    if isinstance(X_single, (list, np.ndarray)):
        X_single = np.array(X_single).reshape(1, -1)
    elif isinstance(X_single, pd.Series):
        X_single = X_single.values.reshape(1, -1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_single)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X_single.shape[1])]

    contributions = {
        name: float(val)
        for name, val in zip(feature_names, shap_values[0])
    }

    # Sort by absolute contribution
    contributions = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return {
        "contributions": contributions,
        "base_value": float(explainer.expected_value) if np.isscalar(explainer.expected_value)
                       else float(explainer.expected_value[1]),
    }


def plot_shap_summary(shap_result: dict, save_path: str = None) -> str:
    """
    Plot SHAP summary (beeswarm) plot and save.

    Args:
        shap_result: Output from get_shap_explanations()
        save_path: Path to save the plot

    Returns:
        Path to saved image
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_result["shap_values"],
        shap_result["X_sample"],
        show=False,
        plot_size=None,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        return save_path

    plt.close("all")
    return None


def plot_shap_bar(shap_result: dict, save_path: str = None, top_n: int = 15) -> str:
    """
    Plot SHAP feature importance bar chart.

    Returns:
        Path to saved image
    """
    importances = shap_result["feature_importance"]
    top_features = dict(list(importances.items())[:top_n])

    features = list(reversed(top_features.keys()))
    values = list(reversed(top_features.values()))

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))
    ax.barh(features, values, color="#42A5F5", edgecolor="none", height=0.6)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("SHAP Feature Importance", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return None
