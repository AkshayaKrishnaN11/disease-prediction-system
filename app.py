"""
🏥 Disease Prediction & Medical Diagnosis System
Streamlit Application — All-in-one frontend for disease prediction.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import io
import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (
    SAVED_MODELS_DIR, DISEASE_CONFIGS, SUPPORTED_DISEASES,
    CNN_PARAMS,
)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.8rem;
        color: #333;
    }
    .metric-card p {
        margin: 0;
        font-size: 0.85rem;
        color: #666;
    }
    .result-positive {
        background: linear-gradient(135deg, #ff6b6b22, #ee535322);
        border: 1px solid #ff6b6b55;
        border-radius: 12px;
        padding: 1.5rem;
    }
    .result-negative {
        background: linear-gradient(135deg, #51cf6622, #2ecc7122);
        border: 1px solid #51cf6655;
        border-radius: 12px;
        padding: 1.5rem;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Helper: Load Tabular Model
# ──────────────────────────────────────────────
@st.cache_resource
def load_tabular_model(disease: str):
    """Load a trained tabular model from disk."""
    disease_dir = SAVED_MODELS_DIR / disease
    for model_file, model_type in [
        ("best_model.pkl", "best"),
        ("xgboost_model.pkl", "xgboost"),
        ("random_forest_model.pkl", "random_forest"),
    ]:
        model_path = disease_dir / model_file
        if model_path.exists():
            model = joblib.load(model_path)
            if model_type == "best":
                model_type = "xgboost" if "XGB" in type(model).__name__ else "random_forest"
            return model, model_type
    return None, None


@st.cache_resource
def load_cnn_model():
    """Load the trained ResNet-50 CNN model."""
    import torch
    model_path = SAVED_MODELS_DIR / "chest_xray" / "resnet50_pneumonia.pth"
    if not model_path.exists():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.models.cnn_trainer_utils import build_resnet50_for_inference
    model = build_resnet50_for_inference(model_path, device)
    return model


def load_disease_metrics(disease: str):
    """Load saved metrics.json for a disease."""
    metrics_path = SAVED_MODELS_DIR / disease / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def get_available_models():
    """Check which models are trained and available."""
    available = []
    for disease in SUPPORTED_DISEASES:
        disease_dir = SAVED_MODELS_DIR / disease
        if any((disease_dir / f).exists() for f in ["best_model.pkl", "xgboost_model.pkl"]):
            available.append(disease)
    if (SAVED_MODELS_DIR / "chest_xray" / "resnet50_pneumonia.pth").exists():
        available.append("chest_xray")
    return available


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "🔬 Disease Prediction", "🫁 X-Ray Analysis", "🎯 Train Models", "📊 Model Metrics"],
        label_visibility="collapsed",
    )
    st.divider()
    available = get_available_models()
    st.markdown(f"**Models Available:** {len(available)}")
    for m in available:
        name = DISEASE_CONFIGS.get(m, {}).get("display_name", m)
        st.markdown(f"  ✅ {name}")
    if not available:
        st.markdown("  ⚠️ No models trained yet")


# ══════════════════════════════════════════════
# PAGE: Home
# ══════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<p class="main-header">Disease Prediction & Medical Diagnosis System</p>', unsafe_allow_html=True)
    st.markdown(
        "An ML-powered system for predicting diseases from patient data and chest X-rays. "
        "Built with **XGBoost**, **Random Forest**, and **ResNet-50 CNN**."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Diseases Supported", len(SUPPORTED_DISEASES))
    with col2:
        st.metric("Models Trained", len(available))
    with col3:
        st.metric("Tabular Models", "XGBoost + RF")
    with col4:
        st.metric("CNN Model", "ResNet-50")

    st.divider()

    st.subheader("Features")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.markdown("""
        **🔬 Tabular Disease Prediction**
        - Diabetes (Pima Indians Dataset)
        - Heart Disease (Cleveland Dataset)
        - Chronic Kidney Disease (UCI Dataset)
        - Liver Disease (Indian Patient Dataset)
        - Breast Cancer (Wisconsin Dataset)
        """)
    with fcol2:
        st.markdown("""
        **🫁 Medical Imaging**
        - Chest X-Ray Pneumonia Detection
        - ResNet-50 Transfer Learning
        - Grad-CAM Heatmap Visualization

        **📊 Explainability**
        - SHAP Feature Contributions (Tabular)
        - Grad-CAM Activation Maps (X-Ray)
        """)

    st.divider()

    st.subheader("Quick Start")
    st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python -m data.download_datasets

# 3. Train models
python -m train.train_tabular --disease all
python -m train.train_cnn --epochs 15

# 4. Run the app
streamlit run app.py
    """, language="bash")


# ══════════════════════════════════════════════
# PAGE: Disease Prediction
# ══════════════════════════════════════════════
elif page == "🔬 Disease Prediction":
    st.markdown('<p class="main-header">Disease Prediction</p>', unsafe_allow_html=True)
    st.markdown("Enter patient vitals and lab reports to get a disease prediction with explainability.")

    # Disease selector
    disease_key = st.selectbox(
        "Select Disease",
        SUPPORTED_DISEASES,
        format_func=lambda x: DISEASE_CONFIGS[x]["display_name"],
    )

    cfg = DISEASE_CONFIGS[disease_key]
    features = cfg["feature_columns"]

    # Check if model exists
    model, model_type = load_tabular_model(disease_key)
    if model is None:
        st.warning(f"⚠️ No trained model found for **{cfg['display_name']}**. "
                   f"Run: `python -m train.train_tabular --disease {disease_key}`")
        st.stop()

    st.info(f"Using **{model_type.upper()}** model for {cfg['display_name']}")

    # Feature input form
    st.subheader("Patient Features")
    num_cols = 3
    feature_values = {}

    cols = st.columns(num_cols)
    for i, feat in enumerate(features):
        with cols[i % num_cols]:
            feature_values[feat] = st.number_input(
                feat.replace("_", " ").title(),
                value=0.0,
                step=0.1,
                key=f"feat_{disease_key}_{feat}",
            )

    # Predict
    if st.button("🔬 Predict", type="primary", width="stretch"):
        X = pd.DataFrame([feature_values])

        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        prob_positive = float(probabilities[1])
        confidence = float(max(probabilities))
        label = cfg["positive_label"] if prediction == 1 else cfg["negative_label"]

        # Display result
        st.divider()
        st.subheader("Prediction Result")

        result_class = "result-positive" if prediction == 1 else "result-negative"
        emoji = "🔴" if prediction == 1 else "🟢"

        st.markdown(f"""
        <div class="{result_class}">
            <h2>{emoji} {label}</h2>
            <p><strong>Confidence:</strong> {confidence:.1%} &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Probability (positive):</strong> {prob_positive:.4f} &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Model:</strong> {model_type}</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar
        st.progress(prob_positive, text=f"Probability: {prob_positive:.1%}")

        # SHAP explanation
        st.subheader("🔍 SHAP Explanation")
        try:
            from src.explainability.shap_explainer import explain_single_prediction
            result = explain_single_prediction(model, X, feature_names=features)
            contributions = result["contributions"]

            # Bar chart of contributions
            shap_df = pd.DataFrame(
                list(contributions.items()),
                columns=["Feature", "SHAP Value"]
            ).sort_values("SHAP Value", key=abs, ascending=True)

            st.bar_chart(shap_df.set_index("Feature"), horizontal=True)

            with st.expander("Raw SHAP Values"):
                st.dataframe(
                    pd.DataFrame(contributions.items(), columns=["Feature", "Impact"]),
                    width="stretch",
                )
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")


# ══════════════════════════════════════════════
# PAGE: X-Ray Analysis
# ══════════════════════════════════════════════
elif page == "🫁 X-Ray Analysis":
    st.markdown('<p class="main-header">Chest X-Ray Analysis</p>', unsafe_allow_html=True)
    st.markdown("Upload a chest X-ray image to detect pneumonia using a ResNet-50 CNN.")

    cnn_model = load_cnn_model()
    if cnn_model is None:
        st.warning("⚠️ CNN model not found. Run: `python -m train.train_cnn`")
        st.stop()

    uploaded = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        import cv2
        import torch
        from src.preprocessing.image import get_val_transforms
        from src.explainability.gradcam import GradCAM

        # Decode image
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if original is None:
            st.error("Could not read the image. Please upload a valid JPEG/PNG.")
            st.stop()
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_rgb, caption="Uploaded X-Ray", width="stretch")

        # Preprocess
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = get_val_transforms()
        augmented = transform(image=original_rgb)
        input_tensor = augmented["image"].unsqueeze(0).to(device)

        # Predict
        cnn_model.eval()
        with torch.no_grad():
            output = cnn_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            prob_pneumonia = float(probs[0][1])
            confidence = float(probs[0].max())

        label = "Pneumonia" if prediction == 1 else "Normal"
        emoji = "🔴" if prediction == 1 else "🟢"
        result_class = "result-positive" if prediction == 1 else "result-negative"

        st.markdown(f"""
        <div class="{result_class}">
            <h2>{emoji} {label}</h2>
            <p><strong>Confidence:</strong> {confidence:.1%} &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Pneumonia Probability:</strong> {prob_pneumonia:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Grad-CAM
        st.subheader("🔍 Grad-CAM Visualization")
        try:
            gradcam = GradCAM(cnn_model, target_layer_name="layer4")
            heatmap = gradcam.generate(input_tensor, target_class=prediction)
            overlay = gradcam.overlay_on_image(original_rgb, heatmap)

            with col2:
                st.image(overlay, caption="Grad-CAM Heatmap Overlay", width="stretch")
        except Exception as e:
            st.warning(f"Grad-CAM visualization unavailable: {e}")


# ══════════════════════════════════════════════
# PAGE: Train Models
# ══════════════════════════════════════════════
elif page == "🎯 Train Models":
    st.markdown('<p class="main-header">Train Models</p>', unsafe_allow_html=True)
    st.markdown("Train ML models on the disease datasets.")

    tab1, tab2 = st.tabs(["📋 Tabular Models", "🫁 CNN (X-Ray)"])

    with tab1:
        st.subheader("Tabular Model Training")
        st.markdown("Trains both **XGBoost** and **Random Forest** models, compares them, and saves the best.")

        disease_to_train = st.multiselect(
            "Select diseases to train",
            SUPPORTED_DISEASES,
            default=SUPPORTED_DISEASES,
            format_func=lambda x: DISEASE_CONFIGS[x]["display_name"],
        )
        skip_shap = st.checkbox("Skip SHAP explanations (faster)", value=False)

        if st.button("🚀 Start Tabular Training", type="primary"):
            from train.train_tabular import train_single_disease

            progress = st.progress(0)
            for i, disease in enumerate(disease_to_train):
                cfg = DISEASE_CONFIGS[disease]
                st.write(f"**Training {cfg['display_name']}...**")
                try:
                    with st.spinner(f"Training {cfg['display_name']}..."):
                        result = train_single_disease(disease, skip_shap=skip_shap)
                    if result:
                        xgb = result.get("xgboost", {})
                        rf = result.get("random_forest", {})
                        st.success(
                            f"✅ {cfg['display_name']} — "
                            f"XGB AUC: {xgb.get('roc_auc', 0):.4f}, "
                            f"RF AUC: {rf.get('roc_auc', 0):.4f}"
                        )
                    else:
                        st.warning(f"⚠️ {cfg['display_name']} — Dataset not found")
                except Exception as e:
                    st.error(f"❌ {cfg['display_name']}: {e}")
                progress.progress((i + 1) / len(disease_to_train))

            st.balloons()
            st.cache_resource.clear()

    with tab2:
        st.subheader("CNN Training (Chest X-Ray)")
        st.markdown("Trains a **ResNet-50** model for pneumonia detection using transfer learning.")

        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=CNN_PARAMS["epochs"])
            batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=CNN_PARAMS["batch_size"])
        with col2:
            lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=CNN_PARAMS["learning_rate"], format="%.6f")
            num_workers = st.number_input("Num Workers", min_value=0, max_value=8, value=0)

        if st.button("🚀 Start CNN Training", type="primary"):
            from train.train_cnn import train_cnn
            with st.spinner("Training CNN... this may take a while."):
                try:
                    metrics = train_cnn(
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=lr,
                        num_workers=num_workers,
                    )
                    if metrics:
                        st.success(
                            f"✅ CNN Training Complete — "
                            f"Accuracy: {metrics['accuracy']:.4f}, "
                            f"F1: {metrics['f1_score']:.4f}"
                        )
                    else:
                        st.warning("⚠️ Training data not found. Download the chest X-ray dataset first.")
                except Exception as e:
                    st.error(f"❌ CNN Training failed: {e}")
            st.cache_resource.clear()


# ══════════════════════════════════════════════
# PAGE: Model Metrics
# ══════════════════════════════════════════════
elif page == "📊 Model Metrics":
    st.markdown('<p class="main-header">Model Performance Metrics</p>', unsafe_allow_html=True)
    st.markdown("View evaluation metrics and plots for trained models.")

    all_diseases = SUPPORTED_DISEASES + ["chest_xray"]
    disease_key = st.selectbox(
        "Select Disease / Model",
        all_diseases,
        format_func=lambda x: DISEASE_CONFIGS.get(x, {}).get("display_name", "Chest X-Ray (CNN)"),
    )

    metrics = load_disease_metrics(disease_key)
    if metrics is None:
        st.warning(f"⚠️ No metrics found for **{disease_key}**. Train the model first.")
        st.stop()

    # For tabular models, metrics has "xgboost" and "random_forest" sub-keys
    if "xgboost" in metrics or "random_forest" in metrics:
        # Tabular model metrics
        for model_name in ["xgboost", "random_forest"]:
            m = metrics.get(model_name)
            if m is None:
                continue
            st.subheader(f"{'🚀 XGBoost' if model_name == 'xgboost' else '🌲 Random Forest'}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
            with col2:
                st.metric("F1 Score", f"{m.get('f1_score', 0):.4f}")
            with col3:
                st.metric("ROC AUC", f"{m.get('roc_auc', 0):.4f}" if m.get('roc_auc') else "N/A")
            with col4:
                cv = m.get('cv_auc_mean')
                st.metric("CV AUC", f"{cv:.4f}" if cv else "N/A")

            # Confusion matrix
            cm = m.get("confusion_matrix")
            if cm:
                st.markdown("**Confusion Matrix**")
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Negative", "Actual Positive"],
                    columns=["Predicted Negative", "Predicted Positive"],
                )
                st.dataframe(cm_df, width="stretch")

            st.divider()
    else:
        # CNN or flat metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        with col2:
            st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
        with col3:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        with col4:
            auc = metrics.get('roc_auc')
            st.metric("ROC AUC", f"{auc:.4f}" if auc else "N/A")

        cm = metrics.get("confusion_matrix")
        if cm:
            st.markdown("**Confusion Matrix**")
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"],
            )
            st.dataframe(cm_df, width="stretch")

    # Display saved plots
    plots_dir = SAVED_MODELS_DIR / disease_key / "plots"
    if plots_dir.exists():
        plot_files = sorted(plots_dir.glob("*.png"))
        if plot_files:
            st.subheader("📈 Visualizations")
            for pf in plot_files:
                st.image(str(pf), caption=pf.stem.replace("_", " ").title(), width="stretch")
