import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

// ── Tabular Prediction ──────────────────────────
export interface TabularPredictionRequest {
  disease: string;
  features: Record<string, number>;
}

export interface TabularPredictionResponse {
  disease: string;
  prediction: number;
  prediction_label: string;
  probability: number;
  confidence: number;
  model_type: string;
  shap_explanation: Record<string, number> | null;
}

export async function predictTabular(data: TabularPredictionRequest): Promise<TabularPredictionResponse> {
  const res = await api.post("/predict/tabular", data);
  return res.data;
}

// ── X-Ray Prediction ────────────────────────────
export interface XRayPredictionResponse {
  prediction: number;
  prediction_label: string;
  probability: number;
  confidence: number;
  gradcam_heatmap: string | null;
}

export async function predictXRay(file: File): Promise<XRayPredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await api.post("/predict/xray", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}

// ── Diseases ────────────────────────────────────
export interface DiseaseInfo {
  display_name: string;
  required_features: string[];
  positive_label: string;
  negative_label: string;
}

export async function getDiseases(): Promise<Record<string, DiseaseInfo>> {
  const res = await api.get("/predict/diseases");
  return res.data.diseases;
}

// ── Metrics ─────────────────────────────────────
export interface ModelMetrics {
  accuracy: number;
  f1_score: number;
  precision: number;
  recall: number;
  roc_auc: number | null;
  confusion_matrix: number[][];
  cv_auc_mean?: number;
  cv_auc_std?: number;
  feature_importance?: Record<string, number>;
  classification_report?: Record<string, any>;
  training_time?: number;
  shap_feature_importance?: Record<string, number>;
}

export async function getMetrics(disease: string): Promise<{ disease: string; metrics: Record<string, ModelMetrics> }> {
  const res = await api.get(`/metrics/${disease}`);
  return res.data;
}

export async function getAllMetrics(): Promise<{ total_models: number; diseases: Record<string, any> }> {
  const res = await api.get("/metrics/");
  return res.data;
}

// ── Health ──────────────────────────────────────
export interface HealthResponse {
  status: string;
  version: string;
  models_loaded: string[];
  gpu_available: boolean;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await api.get("/health");
  return res.data;
}
