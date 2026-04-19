"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useMutation } from "@tanstack/react-query";
import { predictXRay, XRayPredictionResponse } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import {
  Upload,
  ScanLine,
  AlertCircle,
  CheckCircle2,
  Loader2,
  ImageIcon,
  X,
} from "lucide-react";

export default function XRayPage() {
  const { addPrediction } = useAppStore();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<XRayPredictionResponse | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/jpeg": [".jpg", ".jpeg"], "image/png": [".png"] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
  });

  const mutation = useMutation({
    mutationFn: (file: File) => predictXRay(file),
    onSuccess: (data) => {
      setResult(data);
      addPrediction({
        id: Date.now().toString(),
        timestamp: new Date(),
        type: "xray",
        result: data,
      });
    },
  });

  const handleAnalyze = () => {
    if (selectedFile) {
      mutation.mutate(selectedFile);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold gradient-text">X-Ray Analysis</h1>
        <p className="text-muted-foreground mt-1">
          Upload a chest X-ray for AI-powered pneumonia detection
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload Area */}
        <div className="space-y-4">
          {/* Dropzone */}
          <div
            {...getRootProps()}
            className={cn(
              "card-elevated rounded-2xl p-8 text-center cursor-pointer transition-all duration-300",
              "border-2 border-dashed",
              isDragActive
                ? "border-primary bg-primary/5 glow"
                : "border-border hover:border-primary/50",
              selectedFile && "border-solid"
            )}
          >
            <input {...getInputProps()} />

            {preview ? (
              <div className="relative">
                <img
                  src={preview}
                  alt="X-Ray Preview"
                  className="max-h-80 mx-auto rounded-xl object-contain"
                />
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    clearSelection();
                  }}
                  className="absolute top-2 right-2 h-8 w-8 rounded-full bg-background/80 flex items-center justify-center hover:bg-destructive transition-colors"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ) : (
              <div className="py-8">
                <Upload className="h-12 w-12 text-muted-foreground/40 mx-auto mb-4" />
                <p className="text-sm font-medium">
                  {isDragActive
                    ? "Drop your X-ray here..."
                    : "Drag & drop a chest X-ray image"}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  or click to browse · JPEG/PNG · Max 10MB
                </p>
              </div>
            )}
          </div>

          {selectedFile && (
            <div className="flex items-center gap-3 text-sm">
              <ImageIcon className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground truncate">
                {selectedFile.name} ({(selectedFile.size / 1024).toFixed(0)} KB)
              </span>
            </div>
          )}

          {/* Analyze Button */}
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || mutation.isPending}
            className={cn(
              "w-full flex items-center justify-center gap-2 rounded-xl px-6 py-3",
              "bg-accent text-accent-foreground font-semibold text-sm",
              "hover:bg-accent/90 transition-all",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              "focus:ring-2 focus:ring-accent/50 focus:outline-none"
            )}
          >
            {mutation.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Analyzing X-Ray...
              </>
            ) : (
              <>
                <ScanLine className="h-4 w-4" />
                Analyze X-Ray
              </>
            )}
          </button>

          {mutation.isError && (
            <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 rounded-lg p-3">
              <AlertCircle className="h-4 w-4" />
              <span>
                {(mutation.error as any)?.response?.data?.detail ||
                  "Analysis failed. Is the API running?"}
              </span>
            </div>
          )}
        </div>

        {/* Results */}
        <div className="space-y-4">
          {result && (
            <>
              {/* Diagnosis */}
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
                    <h3 className="text-xl font-bold">
                      {result.prediction_label}
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      ResNet-50 Transfer Learning
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">
                      Pneumonia Probability
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

              {/* Grad-CAM Heatmap */}
              {result.gradcam_heatmap && (
                <div className="card-elevated rounded-2xl p-6 animate-fade-in">
                  <h3 className="text-lg font-semibold mb-1">
                    Grad-CAM Heatmap
                  </h3>
                  <p className="text-xs text-muted-foreground mb-4">
                    Regions highlighted by the AI model for its decision
                  </p>
                  <img
                    src={`data:image/png;base64,${result.gradcam_heatmap}`}
                    alt="Grad-CAM Heatmap"
                    className="w-full rounded-xl"
                  />
                </div>
              )}
            </>
          )}

          {!result && !mutation.isPending && (
            <div className="card-elevated rounded-2xl p-12 text-center">
              <ScanLine className="h-12 w-12 text-muted-foreground/30 mx-auto mb-4" />
              <p className="text-muted-foreground">
                Upload an X-ray image and click Analyze
              </p>
              <p className="text-xs text-muted-foreground mt-2">
                The AI will detect pneumonia and show Grad-CAM heatmap
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
