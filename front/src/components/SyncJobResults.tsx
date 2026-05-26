import { useEffect } from "react";
import { Link, useParams } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";

export function SyncJobResults({ children }: { children: React.ReactNode }) {
  const { jobId } = useParams<{ jobId: string }>();
  const { result, loadResults, loading, error } = useAnalysis();

  useEffect(() => {
    if (!jobId) return;
    if (result?.job.id === jobId) return;
    void loadResults(jobId);
  }, [jobId, result?.job.id, loadResults]);

  if (!jobId) return null;

  if (loading && result?.job.id !== jobId) {
    return (
      <div className="empty-state">
        <div className="spinner" style={{ margin: "0 auto 1rem" }} />
        <p>Chargement des résultats…</p>
      </div>
    );
  }

  if (error && result?.job.id !== jobId) {
    return (
      <div className="empty-state">
        <h2>Impossible de charger l&apos;analyse</h2>
        <p style={{ color: "var(--text-muted)", maxWidth: "32rem", margin: "0.5rem auto" }}>{error}</p>
        <p style={{ fontSize: "0.9rem", color: "var(--text-muted)" }}>
          Vérifiez que l&apos;API tourne (<code>./run-dev.sh</code>) et que le job existe.
        </p>
        <Link to="/" className="btn btn-primary" style={{ marginTop: "1rem", display: "inline-flex" }}>
          Nouvelle analyse
        </Link>
      </div>
    );
  }

  return <>{children}</>;
}
