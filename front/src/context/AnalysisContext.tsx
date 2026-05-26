import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import type { AnalysisResult } from "../types";
import { getAnalysisResults } from "../api/client";

type Ctx = {
  result: AnalysisResult | null;
  jobId: string | null;
  loading: boolean;
  error: string | null;
  loadResults: (id: string) => Promise<void>;
  clear: () => void;
};

const AnalysisContext = createContext<Ctx | null>(null);

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadResults = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);
    setJobId(id);
    try {
      for (let i = 0; i < 120; i++) {
        try {
          setResult(await getAnalysisResults(id));
          return;
        } catch (e) {
          const msg = e instanceof Error ? e.message : "";
          if (msg === "ANALYSIS_IN_PROGRESS") {
            await new Promise((r) => setTimeout(r, 1000));
            continue;
          }
          throw e;
        }
      }
      throw new Error("L'analyse prend trop de temps — réessayez dans quelques instants.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const clear = useCallback(() => {
    setResult(null);
    setJobId(null);
    setError(null);
  }, []);

  const value = useMemo(
    () => ({ result, jobId, loading, error, loadResults, clear }),
    [result, jobId, loading, error, loadResults, clear]
  );

  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>;
}

export function useAnalysis() {
  const c = useContext(AnalysisContext);
  if (!c) throw new Error("useAnalysis");
  return c;
}
