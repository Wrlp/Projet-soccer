import type { AnalysisResult } from "../types";

const LABELS = ["Goal", "Yellow card", "Corner", "Foul", "Substitution", "Shots on target"];

function rnd(seed: number) {
  let s = seed;
  return () => ((s = (s * 16807) % 2147483647) - 1) / 2147483646;
}

export function buildMockAnalysis(jobId: string, videoName: string): AnalysisResult {
  const r = rnd(jobId.length);
  const predictions = Array.from({ length: 20 }, () => ({
    timestamp: Math.floor(r() * 2700),
    half: (r() > 0.5 ? 2 : 1) as 1 | 2,
    label: LABELS[Math.floor(r() * LABELS.length)],
    confidence: 0.55 + r() * 0.4,
  })).sort((a, b) => a.timestamp - b.timestamp);

  const counts: Record<string, number> = {};
  predictions.forEach((p) => (counts[p.label] = (counts[p.label] ?? 0) + 1));

  return {
    job: { id: jobId, status: "completed", videoName, createdAt: new Date().toISOString() },
    predictions,
    groundTruth: [],
    metrics: { precision: 0.72, recall: 0.68, f1: 0.7, accuracy: 0.74, meanTemporalErrorSec: 0, withinTolerancePct: 0, toleranceSec: 5 },
    classMetrics: Object.entries(counts).map(([label, count]) => ({ label, count, map: 50 + count })),
    confusionMatrix: { labels: LABELS.slice(0, 4), matrix: LABELS.slice(0, 4).map((_, i) => LABELS.slice(0, 4).map((_, j) => (i === j ? 5 : 1))) },
    trainingCurves: {
      epochs: Array.from({ length: 50 }, (_, i) => i + 1),
      trainLoss: Array.from({ length: 50 }, (_, i) => 1.8 * Math.exp(-0.07 * i)),
      valLoss: Array.from({ length: 50 }, (_, i) => 1.9 * Math.exp(-0.06 * i) + 0.1),
      valMap: Array.from({ length: 50 }, (_, i) => 60 * (1 - Math.exp(-0.08 * i))),
    },
    hyperparams: [
      { learningRate: 1e-3, batchSize: 16, map: 58.6 },
      { learningRate: 1e-4, batchSize: 8, map: 42.1 },
    ],
    thresholdCurve: {
      thresholds: Array.from({ length: 20 }, (_, i) => 0.1 + i * 0.04),
      precision: Array.from({ length: 20 }, (_, i) => 0.4 + 0.55 * (0.1 + i * 0.04)),
      recall: Array.from({ length: 20 }, (_, j) => 0.95 - 0.75 * (0.1 + j * 0.04)),
    },
    classDistribution: Object.entries(counts).map(([label, count]) => ({ label, count })),
    meta: {
      framesAnalyzed: 120,
      durationSeconds: 60,
      durationMinutes: 1,
      fps: 2,
      isExcerpt: true,
      analysisMode: "excerpt",
      segmentLabel: "Extrait vidéo",
    },
  };
}
