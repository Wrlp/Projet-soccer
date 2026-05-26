import type { AnalysisParams, AnalysisResult, UploadResponse } from "../types";
import { buildMockAnalysis } from "./mockData";

const USE_MOCK = import.meta.env.VITE_USE_MOCK === "true";
const API = import.meta.env.VITE_API_URL ?? "/api";

export function getJobVideoUrl(jobId: string): string {
  return `${API}/jobs/${jobId}/video`;
}

export async function uploadAndAnalyze(file: File, params: AnalysisParams): Promise<UploadResponse> {
  if (USE_MOCK) {
    await new Promise((r) => setTimeout(r, 800));
    return { jobId: `job-${Date.now().toString(36)}` };
  }
  const form = new FormData();
  form.append("video", file);
  form.append("threshold", String(params.threshold));
  form.append("context_frames", String(params.contextFrames));
  if (params.half && params.half !== "auto") form.append("half", String(params.half));
  const res = await fetch(`${API}/analyze`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getJobStatus(jobId: string) {
  if (USE_MOCK) return { status: "completed", progress: 100 };
  const res = await fetch(`${API}/jobs/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getAnalysisResults(jobId: string): Promise<AnalysisResult> {
  if (USE_MOCK) {
    await new Promise((r) => setTimeout(r, 400));
    return buildMockAnalysis(jobId, "demo.mkv");
  }
  const res = await fetch(`${API}/jobs/${jobId}/results`);
  if (res.status === 202) throw new Error("ANALYSIS_IN_PROGRESS");
  if (!res.ok) {
    const t = await res.text();
    let detail = t;
    try {
      const j = JSON.parse(t) as { detail?: string };
      if (j.detail) detail = j.detail;
    } catch {
      /* corps non-JSON */
    }
    throw new Error(detail || `Erreur HTTP ${res.status}`);
  }
  return res.json();
}

export async function pollUntilComplete(jobId: string, onProgress?: (p: number) => void) {
  for (let i = 0; i < 120; i++) {
    const { status, progress, error } = await getJobStatus(jobId);
    onProgress?.(progress ?? Math.min(95, (i + 1) * 2));
    if (status === "completed") {
      onProgress?.(100);
      return;
    }
    if (status === "failed") throw new Error(error ?? "Échec");
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error("Délai dépassé");
}
