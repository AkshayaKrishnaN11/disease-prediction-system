import { create } from "zustand";
import { TabularPredictionResponse, XRayPredictionResponse } from "./api";

// ── Disease Feature Defaults (for form fields) ──
export const DISEASE_FEATURES: Record<string, { label: string; fields: { name: string; label: string; min: number; max: number; step: number; placeholder: string }[] }> = {
  diabetes: {
    label: "Diabetes",
    fields: [
      { name: "Pregnancies", label: "Pregnancies", min: 0, max: 20, step: 1, placeholder: "e.g. 6" },
      { name: "Glucose", label: "Glucose (mg/dL)", min: 0, max: 300, step: 1, placeholder: "e.g. 148" },
      { name: "BloodPressure", label: "Blood Pressure (mmHg)", min: 0, max: 200, step: 1, placeholder: "e.g. 72" },
      { name: "SkinThickness", label: "Skin Thickness (mm)", min: 0, max: 100, step: 1, placeholder: "e.g. 35" },
      { name: "Insulin", label: "Insulin (μU/ml)", min: 0, max: 900, step: 1, placeholder: "e.g. 0" },
      { name: "BMI", label: "BMI", min: 0, max: 70, step: 0.1, placeholder: "e.g. 33.6" },
      { name: "DiabetesPedigreeFunction", label: "Diabetes Pedigree", min: 0, max: 3, step: 0.001, placeholder: "e.g. 0.627" },
      { name: "Age", label: "Age", min: 0, max: 120, step: 1, placeholder: "e.g. 50" },
    ],
  },
  heart: {
    label: "Heart Disease",
    fields: [
      { name: "age", label: "Age", min: 0, max: 120, step: 1, placeholder: "e.g. 63" },
      { name: "sex", label: "Sex (1=M, 0=F)", min: 0, max: 1, step: 1, placeholder: "1 or 0" },
      { name: "cp", label: "Chest Pain Type (0-3)", min: 0, max: 3, step: 1, placeholder: "e.g. 3" },
      { name: "trestbps", label: "Resting BP (mmHg)", min: 50, max: 250, step: 1, placeholder: "e.g. 145" },
      { name: "chol", label: "Cholesterol (mg/dL)", min: 100, max: 600, step: 1, placeholder: "e.g. 233" },
      { name: "fbs", label: "Fasting Blood Sugar > 120 (1/0)", min: 0, max: 1, step: 1, placeholder: "1 or 0" },
      { name: "restecg", label: "Resting ECG (0-2)", min: 0, max: 2, step: 1, placeholder: "e.g. 0" },
      { name: "thalach", label: "Max Heart Rate", min: 50, max: 250, step: 1, placeholder: "e.g. 150" },
      { name: "exang", label: "Exercise Angina (1/0)", min: 0, max: 1, step: 1, placeholder: "1 or 0" },
      { name: "oldpeak", label: "ST Depression", min: 0, max: 10, step: 0.1, placeholder: "e.g. 2.3" },
      { name: "slope", label: "ST Slope (0-2)", min: 0, max: 2, step: 1, placeholder: "e.g. 0" },
      { name: "ca", label: "Major Vessels (0-3)", min: 0, max: 3, step: 1, placeholder: "e.g. 0" },
      { name: "thal", label: "Thalassemia (0-3)", min: 0, max: 3, step: 1, placeholder: "e.g. 1" },
    ],
  },
  kidney: {
    label: "Chronic Kidney Disease",
    fields: [
      { name: "age", label: "Age", min: 0, max: 120, step: 1, placeholder: "e.g. 48" },
      { name: "bp", label: "Blood Pressure", min: 50, max: 200, step: 1, placeholder: "e.g. 80" },
      { name: "sg", label: "Specific Gravity", min: 1.0, max: 1.03, step: 0.005, placeholder: "e.g. 1.020" },
      { name: "al", label: "Albumin (0-5)", min: 0, max: 5, step: 1, placeholder: "e.g. 1" },
      { name: "su", label: "Sugar (0-5)", min: 0, max: 5, step: 1, placeholder: "e.g. 0" },
      { name: "bgr", label: "Blood Glucose Random", min: 50, max: 500, step: 1, placeholder: "e.g. 121" },
      { name: "bu", label: "Blood Urea", min: 1, max: 400, step: 1, placeholder: "e.g. 36" },
      { name: "sc", label: "Serum Creatinine", min: 0, max: 80, step: 0.1, placeholder: "e.g. 1.2" },
      { name: "hemo", label: "Hemoglobin", min: 3, max: 20, step: 0.1, placeholder: "e.g. 15.4" },
      { name: "pcv", label: "Packed Cell Volume", min: 10, max: 60, step: 1, placeholder: "e.g. 44" },
      { name: "rc", label: "Red Blood Cell Count", min: 2, max: 8, step: 0.1, placeholder: "e.g. 5.2" },
      { name: "htn", label: "Hypertension (1/0)", min: 0, max: 1, step: 1, placeholder: "1 or 0" },
      { name: "dm", label: "Diabetes Mellitus (1/0)", min: 0, max: 1, step: 1, placeholder: "1 or 0" },
    ],
  },
  liver: {
    label: "Liver Disease",
    fields: [
      { name: "Age", label: "Age", min: 0, max: 120, step: 1, placeholder: "e.g. 65" },
      { name: "Gender", label: "Gender (1=M, 0=F)", min: 0, max: 1, step: 1, placeholder: "1 or 0" },
      { name: "Total_Bilirubin", label: "Total Bilirubin", min: 0, max: 80, step: 0.1, placeholder: "e.g. 0.7" },
      { name: "Direct_Bilirubin", label: "Direct Bilirubin", min: 0, max: 20, step: 0.1, placeholder: "e.g. 0.1" },
      { name: "Alkaline_Phosphotase", label: "Alkaline Phosphatase", min: 50, max: 2200, step: 1, placeholder: "e.g. 187" },
      { name: "Alamine_Aminotransferase", label: "ALT (SGPT)", min: 5, max: 2000, step: 1, placeholder: "e.g. 16" },
      { name: "Aspartate_Aminotransferase", label: "AST (SGOT)", min: 5, max: 5000, step: 1, placeholder: "e.g. 18" },
      { name: "Total_Protiens", label: "Total Proteins", min: 2, max: 10, step: 0.1, placeholder: "e.g. 6.8" },
      { name: "Albumin", label: "Albumin", min: 0, max: 6, step: 0.1, placeholder: "e.g. 3.3" },
      { name: "Albumin_and_Globulin_Ratio", label: "A/G Ratio", min: 0, max: 6, step: 0.01, placeholder: "e.g. 0.9" },
    ],
  },
  breast_cancer: {
    label: "Breast Cancer",
    fields: [
      { name: "radius_mean", label: "Radius Mean", min: 5, max: 30, step: 0.01, placeholder: "e.g. 17.99" },
      { name: "texture_mean", label: "Texture Mean", min: 5, max: 45, step: 0.01, placeholder: "e.g. 10.38" },
      { name: "perimeter_mean", label: "Perimeter Mean", min: 40, max: 200, step: 0.1, placeholder: "e.g. 122.8" },
      { name: "area_mean", label: "Area Mean", min: 100, max: 2600, step: 1, placeholder: "e.g. 1001" },
      { name: "smoothness_mean", label: "Smoothness Mean", min: 0.05, max: 0.2, step: 0.001, placeholder: "e.g. 0.118" },
      { name: "compactness_mean", label: "Compactness Mean", min: 0, max: 0.5, step: 0.001, placeholder: "e.g. 0.278" },
      { name: "concavity_mean", label: "Concavity Mean", min: 0, max: 0.5, step: 0.001, placeholder: "e.g. 0.300" },
      { name: "concave points_mean", label: "Concave Points Mean", min: 0, max: 0.3, step: 0.001, placeholder: "e.g. 0.147" },
      { name: "symmetry_mean", label: "Symmetry Mean", min: 0.1, max: 0.4, step: 0.001, placeholder: "e.g. 0.242" },
      { name: "fractal_dimension_mean", label: "Fractal Dimension Mean", min: 0.04, max: 0.1, step: 0.001, placeholder: "e.g. 0.079" },
    ],
  },
};

// ── Store Types ──────────────────────────────────
interface PredictionResult {
  id: string;
  timestamp: Date;
  type: "tabular" | "xray";
  disease?: string;
  result: TabularPredictionResponse | XRayPredictionResponse;
}

interface AppState {
  // Active disease selection
  activeDiseaseKey: string;
  setActiveDiseaseKey: (key: string) => void;

  // Sidebar
  sidebarOpen: boolean;
  toggleSidebar: () => void;

  // Prediction history
  predictions: PredictionResult[];
  addPrediction: (result: PredictionResult) => void;
  clearPredictions: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeDiseaseKey: "diabetes",
  setActiveDiseaseKey: (key) => set({ activeDiseaseKey: key }),

  sidebarOpen: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  predictions: [],
  addPrediction: (result) => set((s) => ({ predictions: [result, ...s.predictions].slice(0, 50) })),
  clearPredictions: () => set({ predictions: [] }),
}));
