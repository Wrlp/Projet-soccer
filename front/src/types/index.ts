export interface DetectedEvent {
  timestamp: number;
  half: 1 | 2;
  label: string;
  confidence: number;
}

export interface GroundTruthEvent {
  timestamp: number;
  half: 1 | 2;
  label: string;
}

export interface UploadResponse {
  jobId: string;
}

export interface AnalysisParams {
  threshold: number;
  contextFrames: number;
  half?: 1 | 2 | "auto";
  model?: string;
}

export interface GlobalMetrics {
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  meanTemporalErrorSec: number;
  withinTolerancePct: number;
  toleranceSec: number;
  note?: string;
}

export interface AnalysisResult {
  job: { id: string; status: string; videoName: string; createdAt: string };
  predictions: DetectedEvent[];
  groundTruth: GroundTruthEvent[];
  metrics: GlobalMetrics;
  classMetrics: { label: string; map: number; count: number }[];
  confusionMatrix: { labels: string[]; matrix: number[][] };
  trainingCurves: { epochs: number[]; trainLoss: number[]; valLoss: number[]; valMap: number[] };
  hyperparams: { learningRate: number; batchSize: number; map: number }[];
  thresholdCurve: { thresholds: number[]; precision: number[]; recall: number[] };
  classDistribution: { label: string; count: number }[];
  evaluationFigures?: string;
  meta?: {
    framesAnalyzed: number;
    durationSeconds: number;
    durationMinutes: number;
    fps: number;
    isExcerpt: boolean;
    analysisMode: "excerpt" | "half" | "match";
    segmentLabel?: string;
    contextFrames?: number;
    detectionHint?: string;
    maxConfidence?: number;
    usedFallback?: boolean;
    thresholdRequested?: number;
    model?: string;
    windowSeconds?: number;
  };
}

export interface ApiModelInfo {
  id: string;
  name: string;
  description: string;
  available: boolean;
  path: string;
  defaultThreshold: number;
  windowSeconds?: number;
  inferFps?: number;
}

export interface ApiModelsResponse {
  default: string;
  models: ApiModelInfo[];
}
