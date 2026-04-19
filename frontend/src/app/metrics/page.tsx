"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { getMetrics, getAllMetrics, ModelMetrics } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  Activity,
  Heart,
  Shield,
  Zap,
  Stethoscope,
  ScanLine,
  BarChart3,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  PieChart,
  Pie,
} from "recharts";

const diseaseIcons: Record<string, any> = {
  diabetes: Activity,
  heart: Heart,
  kidney: Shield,
  liver: Zap,
  breast_cancer: Stethoscope,
  chest_xray: ScanLine,
};

const diseaseNames: Record<string, string> = {
  diabetes: "Diabetes",
  heart: "Heart Disease",
  kidney: "Kidney Disease",
  liver: "Liver Disease",
  breast_cancer: "Breast Cancer",
  chest_xray: "Pneumonia (X-Ray)",
};

export default function MetricsPage() {
  const searchParams = useSearchParams();
  const initialDisease = searchParams.get("disease") || "diabetes";
  const [selectedDisease, setSelectedDisease] = useState(initialDisease);

  const { data: metricsData, isLoading } = useQuery({
    queryKey: ["metrics", selectedDisease],
    queryFn: () => getMetrics(selectedDisease),
    retry: false,
  });

  // Extract metrics — handle both nested (tabular) and flat (xray) formats
  const allMetrics = metricsData?.metrics || {};
  const xgbMetrics = (allMetrics as any)?.xgboost as ModelMetrics | undefined;
  const rfMetrics = (allMetrics as any)?.random_forest as ModelMetrics | undefined;
  // If no nested model keys, treat the metrics object itself as the data (flat format for X-ray)
  const isFlat = !xgbMetrics && !rfMetrics && (allMetrics as any)?.accuracy !== undefined;
  const bestMetrics = xgbMetrics || rfMetrics || (isFlat ? allMetrics as unknown as ModelMetrics : undefined);

  // Comparison data
  const comparisonData = [];
  if (xgbMetrics && rfMetrics) {
    comparisonData.push(
      { metric: "Accuracy", XGBoost: +(xgbMetrics.accuracy * 100).toFixed(1), RandomForest: +(rfMetrics.accuracy * 100).toFixed(1) },
      { metric: "F1 Score", XGBoost: +(xgbMetrics.f1_score * 100).toFixed(1), RandomForest: +(rfMetrics.f1_score * 100).toFixed(1) },
      { metric: "Precision", XGBoost: +(xgbMetrics.precision * 100).toFixed(1), RandomForest: +(rfMetrics.precision * 100).toFixed(1) },
      { metric: "Recall", XGBoost: +(xgbMetrics.recall * 100).toFixed(1), RandomForest: +(rfMetrics.recall * 100).toFixed(1) },
      { metric: "AUC", XGBoost: +((xgbMetrics.roc_auc || 0) * 100).toFixed(1), RandomForest: +((rfMetrics.roc_auc || 0) * 100).toFixed(1) },
    );
  }

  // Feature importance data
  const featureImportance = xgbMetrics?.shap_feature_importance || rfMetrics?.feature_importance || (allMetrics as any)?.rf_feature_importance;
  const featureData = featureImportance
    ? Object.entries(featureImportance as Record<string, number>)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 12)
        .map(([name, value]) => ({
          name: name.length > 18 ? name.slice(0, 18) + "…" : name,
          value: parseFloat(value.toFixed(4)),
        }))
    : [];

  // Confusion matrix
  const confusionMatrix = bestMetrics?.confusion_matrix;

  // Radar data
  const radarData = bestMetrics
    ? [
        { metric: "Accuracy", value: bestMetrics.accuracy * 100 },
        { metric: "F1", value: bestMetrics.f1_score * 100 },
        { metric: "Precision", value: bestMetrics.precision * 100 },
        { metric: "Recall", value: bestMetrics.recall * 100 },
        { metric: "AUC", value: (bestMetrics.roc_auc || 0) * 100 },
      ]
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold gradient-text">Model Metrics</h1>
        <p className="text-muted-foreground mt-1">
          Performance analysis of trained models
        </p>
      </div>

      {/* Disease Tabs */}
      <div className="flex flex-wrap gap-2">
        {Object.entries(diseaseNames).map(([key, name]) => {
          const Icon = diseaseIcons[key] || BarChart3;
          return (
            <button
              key={key}
              onClick={() => setSelectedDisease(key)}
              className={cn(
                "flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-all",
                key === selectedDisease
                  ? "bg-primary/15 text-primary border border-primary/30 glow"
                  : "bg-secondary/50 text-muted-foreground hover:text-foreground hover:bg-secondary border border-transparent"
              )}
            >
              <Icon className="h-4 w-4" />
              {name}
            </button>
          );
        })}
      </div>

      {isLoading && (
        <div className="card-elevated rounded-2xl p-12 text-center">
          <div className="animate-pulse text-muted-foreground">Loading metrics...</div>
        </div>
      )}

      {!isLoading && !bestMetrics && (
        <div className="card-elevated rounded-2xl p-12 text-center">
          <BarChart3 className="h-12 w-12 text-muted-foreground/30 mx-auto mb-4" />
          <p className="text-muted-foreground">
            No metrics available for {diseaseNames[selectedDisease]}
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Train the model first or start the API server
          </p>
        </div>
      )}

      {bestMetrics && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {[
              { label: "Accuracy", value: bestMetrics.accuracy },
              { label: "F1 Score", value: bestMetrics.f1_score },
              { label: "Precision", value: bestMetrics.precision },
              { label: "Recall", value: bestMetrics.recall },
              { label: "ROC-AUC", value: bestMetrics.roc_auc },
            ].map((m) => (
              <div key={m.label} className="card-elevated rounded-xl p-4 text-center">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
                  {m.label}
                </p>
                <p className="text-2xl font-bold mt-1">
                  {m.value ? (m.value * 100).toFixed(1) + "%" : "N/A"}
                </p>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Model Comparison */}
            {comparisonData.length > 0 && (
              <div className="card-elevated rounded-2xl p-6">
                <h3 className="text-lg font-semibold mb-4">
                  XGBoost vs Random Forest
                </h3>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={comparisonData} margin={{ left: -10 }}>
                    <XAxis dataKey="metric" tick={{ fontSize: 11, fill: "#888" }} />
                    <YAxis tick={{ fontSize: 10, fill: "#888" }} domain={[0, 100]} />
                    <Tooltip
                      contentStyle={{
                        background: "hsl(222, 47%, 8%)",
                        border: "1px solid hsl(217, 33%, 17%)",
                        borderRadius: "8px",
                        fontSize: 12,
                      }}
                    />
                    <Bar dataKey="XGBoost" fill="hsl(199, 89%, 48%)" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="RandomForest" fill="hsl(263, 70%, 58%)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <div className="flex justify-center gap-6 mt-2">
                  <div className="flex items-center gap-2 text-xs">
                    <div className="w-3 h-3 rounded bg-[hsl(199,89%,48%)]" />
                    XGBoost
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <div className="w-3 h-3 rounded bg-[hsl(263,70%,58%)]" />
                    Random Forest
                  </div>
                </div>
              </div>
            )}

            {/* Radar Chart */}
            {radarData.length > 0 && (
              <div className="card-elevated rounded-2xl p-6">
                <h3 className="text-lg font-semibold mb-4">Performance Radar</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="hsl(217, 33%, 20%)" />
                    <PolarAngleAxis
                      dataKey="metric"
                      tick={{ fontSize: 11, fill: "#aaa" }}
                    />
                    <PolarRadiusAxis domain={[0, 100]} tick={false} />
                    <Radar
                      dataKey="value"
                      stroke="hsl(199, 89%, 48%)"
                      fill="hsl(199, 89%, 48%)"
                      fillOpacity={0.2}
                      strokeWidth={2}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Confusion Matrix */}
            {confusionMatrix && (
              <div className="card-elevated rounded-2xl p-6">
                <h3 className="text-lg font-semibold mb-4">Confusion Matrix</h3>
                <div className="flex justify-center">
                  <div className="grid grid-cols-2 gap-2 max-w-xs w-full">
                    <div className="text-center text-xs text-muted-foreground col-span-2 mb-2">
                      <span className="font-medium">Predicted →</span>
                    </div>
                    {confusionMatrix.flat().map((val: number, i: number) => {
                      const isDiagonal = i === 0 || i === 3;
                      return (
                        <div
                          key={i}
                          className={cn(
                            "rounded-xl p-6 text-center font-bold text-2xl",
                            isDiagonal
                              ? "bg-green-500/10 border border-green-500/30 text-green-400"
                              : "bg-red-500/10 border border-red-500/30 text-red-400"
                          )}
                        >
                          {val}
                          <p className="text-[10px] font-normal text-muted-foreground mt-1">
                            {i === 0
                              ? "True Neg"
                              : i === 1
                              ? "False Pos"
                              : i === 2
                              ? "False Neg"
                              : "True Pos"}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* Feature Importance */}
            {featureData.length > 0 && (
              <div className="card-elevated rounded-2xl p-6">
                <h3 className="text-lg font-semibold mb-4">Feature Importance</h3>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={featureData} layout="vertical" margin={{ left: 10 }}>
                    <XAxis type="number" tick={{ fontSize: 10, fill: "#888" }} />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={130}
                      tick={{ fontSize: 11, fill: "#aaa" }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "hsl(222, 47%, 8%)",
                        border: "1px solid hsl(217, 33%, 17%)",
                        borderRadius: "8px",
                        fontSize: 12,
                      }}
                    />
                    <Bar dataKey="value" fill="hsl(199, 89%, 48%)" radius={[0, 4, 4, 0]}>
                      {featureData.map((_, i) => (
                        <Cell
                          key={i}
                          fill={`hsl(${199 + i * 8}, 80%, ${48 + i * 2}%)`}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
