"use client";

import { useQuery } from "@tanstack/react-query";
import { getAllMetrics, getHealth } from "@/lib/api";
import {
  Activity,
  Brain,
  Heart,
  Stethoscope,
  ScanLine,
  Zap,
  TrendingUp,
  Shield,
} from "lucide-react";
import Link from "next/link";

const diseaseIcons: Record<string, any> = {
  diabetes: Activity,
  heart: Heart,
  kidney: Shield,
  liver: Zap,
  breast_cancer: Stethoscope,
  chest_xray: ScanLine,
};

const diseaseColors: Record<string, string> = {
  diabetes: "from-blue-500/20 to-cyan-500/20 border-blue-500/30",
  heart: "from-red-500/20 to-pink-500/20 border-red-500/30",
  kidney: "from-green-500/20 to-emerald-500/20 border-green-500/30",
  liver: "from-yellow-500/20 to-amber-500/20 border-yellow-500/30",
  breast_cancer: "from-purple-500/20 to-violet-500/20 border-purple-500/30",
  chest_xray: "from-indigo-500/20 to-sky-500/20 border-indigo-500/30",
};

const diseaseNames: Record<string, string> = {
  diabetes: "Diabetes",
  heart: "Heart Disease",
  kidney: "Kidney Disease",
  liver: "Liver Disease",
  breast_cancer: "Breast Cancer",
  chest_xray: "Pneumonia (X-Ray)",
};

export default function DashboardPage() {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    retry: false,
  });

  const { data: metricsData } = useQuery({
    queryKey: ["allMetrics"],
    queryFn: getAllMetrics,
    retry: false,
  });

  const isBackendOnline = !!health;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold gradient-text">Dashboard</h1>
        <p className="text-muted-foreground mt-1">
          Disease Prediction & Medical Diagnosis System
        </p>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Backend Status */}
        <div className="card-elevated rounded-2xl p-5">
          <div className="flex items-center gap-3">
            <div
              className={`h-3 w-3 rounded-full ${
                isBackendOnline ? "bg-green-400 animate-pulse" : "bg-red-400"
              }`}
            />
            <span className="text-sm font-medium">
              {isBackendOnline ? "API Online" : "API Offline"}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            {isBackendOnline
              ? `v${health?.version} • ${health?.models_loaded.length || 0} models loaded`
              : "Start FastAPI: python -m api.main"}
          </p>
        </div>

        {/* GPU Status */}
        <div className="card-elevated rounded-2xl p-5">
          <div className="flex items-center gap-3">
            <Zap
              className={`h-5 w-5 ${
                health?.gpu_available ? "text-yellow-400" : "text-muted-foreground"
              }`}
            />
            <span className="text-sm font-medium">
              {health?.gpu_available ? "GPU Active" : "CPU Mode"}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            {health?.gpu_available
              ? "CUDA acceleration enabled"
              : "GPU not detected"}
          </p>
        </div>

        {/* Models Count */}
        <div className="card-elevated rounded-2xl p-5">
          <div className="flex items-center gap-3">
            <Brain className="h-5 w-5 text-primary" />
            <span className="text-sm font-medium">
              {metricsData?.total_models || 0} Models Trained
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            XGBoost + Random Forest + ResNet-50
          </p>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Link href="/predict">
          <div className="card-elevated rounded-2xl p-6 cursor-pointer group">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10 group-hover:bg-primary/20 transition-colors">
                <Stethoscope className="h-6 w-6 text-primary" />
              </div>
              <div>
                <h3 className="font-semibold">Disease Prediction</h3>
                <p className="text-sm text-muted-foreground">
                  Enter patient vitals & lab reports
                </p>
              </div>
              <TrendingUp className="h-5 w-5 text-muted-foreground ml-auto group-hover:text-primary transition-colors" />
            </div>
          </div>
        </Link>

        <Link href="/xray">
          <div className="card-elevated rounded-2xl p-6 cursor-pointer group">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-accent/10 group-hover:bg-accent/20 transition-colors">
                <ScanLine className="h-6 w-6 text-accent" />
              </div>
              <div>
                <h3 className="font-semibold">X-Ray Analysis</h3>
                <p className="text-sm text-muted-foreground">
                  Upload chest X-ray for pneumonia detection
                </p>
              </div>
              <TrendingUp className="h-5 w-5 text-muted-foreground ml-auto group-hover:text-accent transition-colors" />
            </div>
          </div>
        </Link>
      </div>

      {/* Model Performance Grid */}
      {metricsData && metricsData.diseases && (
        <div>
          <h2 className="text-xl font-semibold mb-4">Model Performance</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(metricsData.diseases).map(([key, metrics]: [string, any]) => {
              const Icon = diseaseIcons[key] || Activity;
              const colorClass = diseaseColors[key] || "from-gray-500/20 to-gray-500/20 border-gray-500/30";
              const name = diseaseNames[key] || key;

              // Get best model metrics
              const best = metrics.xgboost || metrics.random_forest || metrics;
              const accuracy = best?.accuracy || best?.test_accuracy || 0;
              const auc = best?.roc_auc || best?.test_roc_auc || 0;

              return (
                <Link href={`/metrics?disease=${key}`} key={key}>
                  <div
                    className={`rounded-2xl p-5 border bg-gradient-to-br ${colorClass} hover:scale-[1.02] transition-all cursor-pointer`}
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <Icon className="h-5 w-5" />
                      <span className="font-medium text-sm">{name}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
                          Accuracy
                        </p>
                        <p className="text-lg font-bold">
                          {(accuracy * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-[10px] text-muted-foreground uppercase tracking-wider">
                          AUC
                        </p>
                        <p className="text-lg font-bold">
                          {auc ? (auc * 100).toFixed(1) + "%" : "N/A"}
                        </p>
                      </div>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      )}

      {/* Getting Started (if backend is offline) */}
      {!isBackendOnline && (
        <div className="rounded-2xl border border-yellow-500/30 bg-yellow-500/5 p-6">
          <h3 className="font-semibold text-yellow-400 mb-3">
            ⚡ Getting Started
          </h3>
          <div className="space-y-2 text-sm text-muted-foreground">
            <p>1. Start the FastAPI backend:</p>
            <code className="block bg-secondary/50 rounded-lg p-2 text-xs font-mono">
              cd d:\CEC\Projects\disease_anti && python -m api.main
            </code>
            <p className="mt-3">
              2. The API will be available at{" "}
              <code className="text-primary">http://localhost:8000</code>
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
