"use client";

import { useAppStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import {
  Stethoscope,
  ScanLine,
  AlertCircle,
  CheckCircle2,
  Trash2,
  Clock,
} from "lucide-react";

export default function HistoryPage() {
  const { predictions, clearPredictions } = useAppStore();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text">History</h1>
          <p className="text-muted-foreground mt-1">
            Past prediction results ({predictions.length})
          </p>
        </div>
        {predictions.length > 0 && (
          <button
            onClick={clearPredictions}
            className="flex items-center gap-2 rounded-xl px-4 py-2 text-sm text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-all"
          >
            <Trash2 className="h-4 w-4" />
            Clear All
          </button>
        )}
      </div>

      {predictions.length === 0 ? (
        <div className="card-elevated rounded-2xl p-12 text-center">
          <Clock className="h-12 w-12 text-muted-foreground/30 mx-auto mb-4" />
          <p className="text-muted-foreground">No prediction history yet</p>
          <p className="text-xs text-muted-foreground mt-1">
            Make a prediction to see it here
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {predictions.map((pred) => {
            const isTabular = pred.type === "tabular";
            const result = pred.result as any;
            const isPositive = result.prediction === 1;

            return (
              <div
                key={pred.id}
                className={cn(
                  "card-elevated rounded-2xl p-5 animate-fade-in",
                  "border",
                  isPositive ? "border-red-500/20" : "border-green-500/20"
                )}
              >
                <div className="flex items-center gap-4">
                  {/* Icon */}
                  <div
                    className={cn(
                      "flex h-10 w-10 items-center justify-center rounded-xl",
                      isTabular ? "bg-primary/10" : "bg-accent/10"
                    )}
                  >
                    {isTabular ? (
                      <Stethoscope className="h-5 w-5 text-primary" />
                    ) : (
                      <ScanLine className="h-5 w-5 text-accent" />
                    )}
                  </div>

                  {/* Details */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      {isPositive ? (
                        <AlertCircle className="h-4 w-4 text-red-400 flex-shrink-0" />
                      ) : (
                        <CheckCircle2 className="h-4 w-4 text-green-400 flex-shrink-0" />
                      )}
                      <span className="font-medium text-sm truncate">
                        {result.prediction_label}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 mt-1">
                      <span className="text-xs text-muted-foreground">
                        {isTabular ? `Disease: ${pred.disease}` : "X-Ray Analysis"}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {isTabular ? `Model: ${result.model_type}` : "ResNet-50"}
                      </span>
                    </div>
                  </div>

                  {/* Probability */}
                  <div className="text-right flex-shrink-0">
                    <p className="text-lg font-bold">
                      {(result.probability * 100).toFixed(1)}%
                    </p>
                    <p className="text-[10px] text-muted-foreground">
                      {new Date(pred.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
