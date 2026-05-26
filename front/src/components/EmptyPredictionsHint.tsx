import { Link } from "react-router-dom";
import type { AnalysisResult } from "../types";

export function EmptyPredictionsHint({ result }: { result: AnalysisResult }) {
  const th = result.meta?.thresholdRequested;
  const maxC = result.meta?.maxConfidence;
  const hint = result.meta?.detectionHint;

  return (
    <div
      className="card"
      style={{
        marginBottom: "1rem",
        borderColor: "var(--accent)",
        background: "rgba(34, 197, 94, 0.06)",
      }}
    >
      <p style={{ margin: 0, lineHeight: 1.5 }}>
        <strong>Aucun événement affiché.</strong> SlowFast produit souvent des scores entre{" "}
        <strong>15 % et 35 %</strong> sur des extraits hors SoccerNet.
        {th !== undefined && (
          <>
            {" "}
            Seuil utilisé : <strong>{(th * 100).toFixed(0)} %</strong>.
          </>
        )}
        {maxC !== undefined && maxC > 0 && (
          <>
            {" "}
            Meilleur score : <strong>{(maxC * 100).toFixed(0)} %</strong>.
          </>
        )}
        {hint && <> {hint}</>}
      </p>
      <p style={{ margin: "0.75rem 0 0", fontSize: "0.9rem", color: "var(--text-muted)" }}>
        Relancez une analyse avec le seuil à <strong>0,15–0,25</strong> (curseur sur la page d&apos;accueil).
      </p>
      <Link to="/" className="btn btn-primary" style={{ marginTop: "1rem", display: "inline-flex" }}>
        Nouvelle analyse
      </Link>
    </div>
  );
}
