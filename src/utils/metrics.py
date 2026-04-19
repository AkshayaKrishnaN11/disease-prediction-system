"""
Evaluation metrics utilities for classification models.
ROC-AUC, F1, Accuracy, Confusion Matrix, and plot generators.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


def evaluate_model(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for class 1 (optional)

    Returns:
        Dictionary with accuracy, f1, precision, recall, auc, confusion_matrix
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path: str = None) -> str:
    """
    Plot and save confusion matrix as an image.

    Returns:
        Path to saved image
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels or ["Negative", "Positive"],
        yticklabels=labels or ["Negative", "Positive"],
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return None


def plot_roc_curve(y_true, y_prob, save_path: str = None) -> str:
    """
    Plot and save ROC curve as an image.

    Returns:
        Path to saved image
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2.5, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#BDBDBD", lw=1.5, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2196F3")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return None


def plot_feature_importance(importances: dict, save_path: str = None, top_n: int = 15) -> str:
    """
    Plot top-N feature importances as a horizontal bar chart.

    Args:
        importances: Dict of {feature_name: importance_value}
        save_path: Path to save the plot image
        top_n: Number of top features to plot

    Returns:
        Path to saved image
    """
    # Sort by importance
    sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features, values = zip(*reversed(sorted_imp))

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.4)))

    colors = ["#EF5350" if v < 0 else "#42A5F5" for v in values]
    ax.barh(features, values, color=colors, edgecolor="none", height=0.6)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return None


def save_metrics_json(metrics: dict, save_path: str):
    """Save metrics dict to a JSON file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
