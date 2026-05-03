"""
Tabular Model Trainer — trains XGBoost + Random Forest for all disease datasets.
Generates SHAP explanations and evaluation plots.

Usage:
    python -m train.train_tabular --disease all
    python -m train.train_tabular --disease diabetes
    python -m train.train_tabular --disease diabetes heart
"""

import argparse
import sys
import time
import joblib
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    SAVED_MODELS_DIR, DISEASE_CONFIGS, SUPPORTED_DISEASES,
    XGB_PARAMS, RF_PARAMS, RANDOM_STATE, CV_FOLDS,
)
from src.preprocessing.tabular import get_preprocessed_data
from src.utils.metrics import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, save_metrics_json,
)
from src.explainability.shap_explainer import (
    get_shap_explanations, plot_shap_summary, plot_shap_bar,
)


def train_single_disease(disease: str, skip_shap: bool = False):
    """
    Train both XGBoost and Random Forest for a single disease.

    Steps:
        1. Load & preprocess data
        2. Train XGBoost with cross-validation
        3. Train Random Forest with cross-validation
        4. Evaluate both on test set
        5. Generate SHAP explanations for best model
        6. Save best model as .pkl
    """
    cfg = DISEASE_CONFIGS[disease]
    print(f"\n{'='*60}")
    print(f"🏥 Training: {cfg['display_name']}")
    print(f"{'='*60}")

    # ─── 1. Load Data ─────────────────────────────
    print("\n📊 Loading and preprocessing data...")
    try:
        X_train, X_test, y_train, y_test = get_preprocessed_data(disease)
    except FileNotFoundError as e:
        print(f"  ⚠ {e}")
        print(f"  → Run: python -m data.download_datasets")
        return None

    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")

    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None

    # Output directory for this disease
    disease_dir = SAVED_MODELS_DIR / disease
    disease_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = disease_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ─── 2. Train XGBoost ─────────────────────────
    print(f"\n🚀 Training XGBoost...")
    start_time = time.time()

    xgb_model = XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_time = time.time() - start_time

    # Predictions
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    # Metrics
    xgb_metrics = evaluate_model(y_test, xgb_pred, xgb_prob)
    xgb_metrics["training_time"] = xgb_time

    # Cross-validation
    cv_scores = cross_val_score(
        XGBClassifier(**XGB_PARAMS), X_train, y_train,
        cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1,
    )
    xgb_metrics["cv_auc_mean"] = float(cv_scores.mean())
    xgb_metrics["cv_auc_std"] = float(cv_scores.std())

    print(f"  ✓ Accuracy:  {xgb_metrics['accuracy']:.4f}")
    print(f"  ✓ F1 Score:  {xgb_metrics['f1_score']:.4f}")
    print(f"  ✓ ROC-AUC:   {xgb_metrics['roc_auc']:.4f}")
    print(f"  ✓ CV AUC:    {xgb_metrics['cv_auc_mean']:.4f} ± {xgb_metrics['cv_auc_std']:.4f}")
    print(f"  ⏱ Training time: {xgb_time:.2f}s")

    # Plots
    plot_confusion_matrix(
        y_test, xgb_pred,
        labels=[cfg["negative_label"], cfg["positive_label"]],
        save_path=str(plots_dir / "xgb_confusion_matrix.png"),
    )
    plot_roc_curve(
        y_test, xgb_prob,
        save_path=str(plots_dir / "xgb_roc_curve.png"),
    )

    # SHAP
    if not skip_shap:
        print("  📊 Generating SHAP explanations...")
        try:
            shap_result = get_shap_explanations(xgb_model, X_test, feature_names)
            plot_shap_summary(
                shap_result, save_path=str(plots_dir / "xgb_shap_summary.png")
            )
            plot_shap_bar(
                shap_result, save_path=str(plots_dir / "xgb_shap_bar.png")
            )
            xgb_metrics["shap_feature_importance"] = shap_result["feature_importance"]
            print("  ✓ SHAP explanations generated")
        except Exception as e:
            print(f"  ⚠ SHAP failed: {e}")

    # Save model
    xgb_model_path = disease_dir / "xgboost_model.pkl"
    joblib.dump(xgb_model, xgb_model_path)

    results["xgboost"] = xgb_metrics

    # ─── 3. Train Random Forest ───────────────────
    print(f"\n🌲 Training Random Forest...")
    start_time = time.time()

    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - start_time

    # Predictions
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    # Metrics
    rf_metrics = evaluate_model(y_test, rf_pred, rf_prob)
    rf_metrics["training_time"] = rf_time

    # Cross-validation
    cv_scores = cross_val_score(
        RandomForestClassifier(**RF_PARAMS), X_train, y_train,
        cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1,
    )
    rf_metrics["cv_auc_mean"] = float(cv_scores.mean())
    rf_metrics["cv_auc_std"] = float(cv_scores.std())

    print(f"  ✓ Accuracy:  {rf_metrics['accuracy']:.4f}")
    print(f"  ✓ F1 Score:  {rf_metrics['f1_score']:.4f}")
    print(f"  ✓ ROC-AUC:   {rf_metrics['roc_auc']:.4f}")
    print(f"  ✓ CV AUC:    {rf_metrics['cv_auc_mean']:.4f} ± {rf_metrics['cv_auc_std']:.4f}")
    print(f"  ⏱ Training time: {rf_time:.2f}s")

    # Plots
    plot_confusion_matrix(
        y_test, rf_pred,
        labels=[cfg["negative_label"], cfg["positive_label"]],
        save_path=str(plots_dir / "rf_confusion_matrix.png"),
    )
    plot_roc_curve(
        y_test, rf_prob,
        save_path=str(plots_dir / "rf_roc_curve.png"),
    )

    # Feature importance from RF
    if feature_names:
        fi = dict(zip(feature_names, rf_model.feature_importances_))
        plot_feature_importance(
            fi, save_path=str(plots_dir / "rf_feature_importance.png")
        )
        rf_metrics["rf_feature_importance"] = fi

    # Save model
    rf_model_path = disease_dir / "random_forest_model.pkl"
    joblib.dump(rf_model, rf_model_path)

    results["random_forest"] = rf_metrics

    # ─── 4. Compare & Save Best ───────────────────
    print(f"\n📈 Model Comparison:")
    xgb_auc = results["xgboost"].get("roc_auc", 0) or 0
    rf_auc = results["random_forest"].get("roc_auc", 0) or 0

    best_model = "xgboost" if xgb_auc >= rf_auc else "random_forest"
    print(f"  XGBoost AUC:       {xgb_auc:.4f}")
    print(f"  Random Forest AUC: {rf_auc:.4f}")
    print(f"  🏆 Best model: {best_model}")

    # Save comparison metrics
    save_metrics_json(results, str(disease_dir / "metrics.json"))

    # Copy best model as 'best_model.pkl'
    best_src = disease_dir / f"{best_model.replace(' ', '_')}_model.pkl"
    best_dst = disease_dir / "best_model.pkl"
    if best_src.exists():
        joblib.dump(joblib.load(best_src), best_dst)

    print(f"\n✅ {cfg['display_name']} training complete!")
    print(f"   Models saved to: {disease_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train tabular disease prediction models")
    parser.add_argument(
        "--disease", nargs="+", default=["all"],
        help=f"Disease(s) to train. Options: {SUPPORTED_DISEASES + ['all']}",
    )
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP explanations")
    args = parser.parse_args()

    diseases = SUPPORTED_DISEASES if "all" in args.disease else args.disease

    print("=" * 60)
    print("🏥 Disease Prediction — Tabular Model Training")
    print("=" * 60)
    print(f"Diseases: {diseases}")
    print(f"Models: XGBoost + Random Forest")
    print(f"Output: {SAVED_MODELS_DIR}")

    all_results = {}
    for disease in diseases:
        if disease not in SUPPORTED_DISEASES:
            print(f"\n⚠ Unknown disease '{disease}', skipping")
            continue
        result = train_single_disease(disease, skip_shap=args.skip_shap)
        if result:
            all_results[disease] = result

    # Final summary
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)

    for disease, result in all_results.items():
        cfg = DISEASE_CONFIGS[disease]
        xgb_auc = result["xgboost"].get("roc_auc", "N/A")
        rf_auc = result["random_forest"].get("roc_auc", "N/A")
        best = "XGB" if (xgb_auc or 0) >= (rf_auc or 0) else "RF"
        xgb_auc_str = f"{xgb_auc:.4f}" if isinstance(xgb_auc, float) else xgb_auc
        rf_auc_str = f"{rf_auc:.4f}" if isinstance(rf_auc, float) else rf_auc
        print(f"  {cfg['display_name']:25s} | XGB: {xgb_auc_str} | RF: {rf_auc_str} | Best: {best}")

    print(f"\n✅ All training complete!")
    print(f"📂 Models: {SAVED_MODELS_DIR}")


if __name__ == "__main__":
    main()
