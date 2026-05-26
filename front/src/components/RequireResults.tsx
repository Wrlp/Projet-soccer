import { Link } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";

export function RequireResults({ children }: { children: React.ReactNode }) {
  const { result, loading, error } = useAnalysis();

  if (loading) {
    return (
      <div className="empty-state">
        <div className="spinner" style={{ margin: "0 auto 1rem" }} />
        <p>Chargement…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="empty-state">
        <h2>Erreur</h2>
        <p style={{ color: "var(--text-muted)" }}>{error}</p>
        <Link to="/" className="btn btn-primary" style={{ marginTop: "1rem", display: "inline-flex" }}>
          Retour à l&apos;accueil
        </Link>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="empty-state">
        <h2>Aucune analyse</h2>
        <Link to="/" className="btn btn-primary" style={{ marginTop: "1rem", display: "inline-flex" }}>
          Analyser une vidéo
        </Link>
      </div>
    );
  }

  return <>{children}</>;
}
