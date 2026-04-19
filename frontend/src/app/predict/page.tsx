"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useMutation } from "@tanstack/react-query";
import { predictTabular, TabularPredictionResponse } from "@/lib/api";
import { useAppStore, DISEASE_FEATURES } from "@/lib/store";
import { cn } from "@/lib/utils";
import {
  Activity,
  Heart,
  Shield,
  Zap,
  Stethoscope,
  ChevronRight,
  AlertCircle,
  CheckCircle2,
  Loader2,
  ArrowRight,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const diseaseIcons: Record<string, any> = {
  diabetes: Activity,
  heart: Heart,
  kidney: Shield,
  liver: Zap,
  breast_cancer: Stethoscope,
};

export default function PredictPage() {
  const { activeDiseaseKey, setActiveDiseaseKey, addPrediction } = useAppStore();
  const [result, setResult] = useState<TabularPredictionResponse | null>(null);

  const disease = DISEASE_FEATURES[activeDiseaseKey];
  if (!disease) return null;

  // Build dynamic Zod schema
  const schemaShape: Record<string, z.ZodNumber> = {};
  disease.fields.forEach((f) => {
    schemaShape[f.name] = z.coerce.number({ invalid_type_error: `${f.label} must be a number` });
  });
  const formSchema = z.object(schemaShape);
  type FormData = z.infer<typeof formSchema>;

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm<FormData>({
    resolver: zodResolver(formSchema),
  });

  const mutation = useMutation({
    mutationFn: (data: FormData) =>
      predictTabular({ disease: activeDiseaseKey, features: data }),
    onSuccess: (data) => {
      setResult(data);
      addPrediction({
        id: Date.now().toString(),
        timestamp: new Date(),
        type: "tabular",
        disease: activeDiseaseKey,
        result: data,
      });
    },
  });

  const onSubmit = (data: FormData) => {
    setResult(null);
    mutation.mutate(data);
  };

  const shapData = result?.shap_explanation
    ? Object.entries(result.shap_explanation)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 10)
        .map(([name, value]) => ({
          name: name.length > 15 ? name.slice(0, 15) + "…" : name,
          value: parseFloat(value.toFixed(4)),
          fill: value > 0 ? "hsl(0, 84%, 60%)" : "hsl(199, 89%, 48%)",
        }))
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold gradient-text">Disease Prediction</h1>
        <p className="text-muted-foreground mt-1">
          Enter patient data to get an AI-powered diagnosis
        </p>
      </div>

      {/* Disease Selector */}
      <div className="flex flex-wrap gap-2">
        {Object.entries(DISEASE_FEATURES).map(([key, d]) => {
          const Icon = diseaseIcons[key] || Activity;
          const isActive = key === activeDiseaseKey;
          return (
            <button
              key={key}
              onClick={() => {
                setActiveDiseaseKey(key);
                setResult(null);
                reset();
              }}
              className={cn(
                "flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-all",
                isActive
                  ? "bg-primary/15 text-primary border border-primary/30 glow"
                  : "bg-secondary/50 text-muted-foreground hover:text-foreground hover:bg-secondary border border-transparent"
              )}
            >
              <Icon className="h-4 w-4" />
              {d.label}
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <div className="card-elevated rounded-2xl p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <ChevronRight className="h-5 w-5 text-primary" />
            Patient Data — {disease.label}
          </h2>

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {disease.fields.map((field) => (
                <div key={field.name}>
                  <label className="text-xs font-medium text-muted-foreground mb-1 block">
                    {field.label}
                  </label>
                  <input
                    type="number"
                    step={field.step}
                    min={field.min}
                    max={field.max}
                    placeholder={field.placeholder}
                    {...register(field.name)}
                    className={cn(
                      "w-full rounded-lg bg-secondary/50 border px-3 py-2 text-sm",
                      "focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50",
                      "placeholder:text-muted-foreground/50 transition-all",
                      errors[field.name]
                        ? "border-destructive"
                        : "border-border"
                    )}
                  />
                  {errors[field.name] && (
                    <p className="text-[10px] text-destructive mt-0.5">
                      {errors[field.name]?.message as string}
                    </p>
                  )}
                </div>
              ))}
            </div>

            <button
              type="submit"
              disabled={mutation.isPending}
              className={cn(
                "w-full flex items-center justify-center gap-2 rounded-xl px-6 py-3",
                "bg-primary text-primary-foreground font-semibold text-sm",
                "hover:bg-primary/90 transition-all",
                "disabled:opacity-50 disabled:cursor-not-allowed",
                "focus:ring-2 focus:ring-primary/50 focus:outline-none"
              )}
            >
              {mutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  Predict
                  <ArrowRight className="h-4 w-4" />
                </>
              )}
            </button>

            {mutation.isError && (
              <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 rounded-lg p-3">
                <AlertCircle className="h-4 w-4" />
                <span>
                  {(mutation.error as any)?.response?.data?.detail ||
                    "Failed to get prediction. Is the API running?"}
                </span>
              </div>
            )}
          </form>
        </div>

        {/* Results Panel */}
        <div className="space-y-4">
          {result && (
            <>
              {/* Prediction Result */}
              <div
                className={cn(
                  "rounded-2xl p-6 border animate-fade-in",
                  result.prediction === 1
                    ? "bg-red-500/5 border-red-500/30"
                    : "bg-green-500/5 border-green-500/30"
                )}
              >
                <div className="flex items-center gap-3 mb-4">
                  {result.prediction === 1 ? (
                    <AlertCircle className="h-8 w-8 text-red-400" />
                  ) : (
                    <CheckCircle2 className="h-8 w-8 text-green-400" />
                  )}
                  <div>
                    <h3 className="text-xl font-bold">{result.prediction_label}</h3>
                    <p className="text-sm text-muted-foreground">
                      Model: {result.model_type}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">
                      Probability
                    </p>
                    <p className="text-2xl font-bold">
                      {(result.probability * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">
                      Confidence
                    </p>
                    <p className="text-2xl font-bold">
                      {(result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                {/* Confidence bar */}
                <div className="mt-4">
                  <div className="h-2 rounded-full bg-secondary overflow-hidden">
                    <div
                      className={cn(
                        "h-full rounded-full transition-all duration-1000",
                        result.prediction === 1 ? "bg-red-400" : "bg-green-400"
                      )}
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* SHAP Explanation */}
              {shapData.length > 0 && (
                <div className="card-elevated rounded-2xl p-6 animate-fade-in">
                  <h3 className="text-lg font-semibold mb-1">
                    Feature Impact (SHAP)
                  </h3>
                  <p className="text-xs text-muted-foreground mb-4">
                    Red = increases risk · Blue = decreases risk
                  </p>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={shapData} layout="vertical" margin={{ left: 10 }}>
                      <XAxis type="number" tick={{ fontSize: 10, fill: "#888" }} />
                      <YAxis
                        type="category"
                        dataKey="name"
                        width={120}
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
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {shapData.map((entry: any, i: number) => (
                          <Cell key={i} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}

          {!result && !mutation.isPending && (
            <div className="card-elevated rounded-2xl p-12 text-center">
              <Stethoscope className="h-12 w-12 text-muted-foreground/30 mx-auto mb-4" />
              <p className="text-muted-foreground">
                Fill in patient data and click Predict to get results
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
