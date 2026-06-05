import { Link } from "react-router-dom";
import { FALLBACK_MODEL_OPTIONS, resolveModel } from "../constants/models";
import type { AnalysisResult } from "../types";

export function EmptyPredictionsHint({ result }: { result: AnalysisResult }) {
  const th = result.meta?.thresholdRequested;
  const maxC = result.meta?.maxConfidence;
  const hint = result.meta?.detectionHint;
  const model = resolveModel(result.meta?.model, FALLBACK_MODEL_OPTIONS);
  const modelName = model?.name ?? result.meta?.model ?? "le modèle";
  const [rangeLow, rangeHigh] = model?.suggestedThresholdRange ?? [0.15, 0.3];

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
        <strong>Aucun événement affiché.</strong>{" "}
        {model?.emptyScoreHint ?? (
          <>Aucune détection de {modelName} n&apos;a dépassé le seuil configuré.</>
        )}
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
        Relancez une analyse avec un seuil plus bas (
        <strong>
          {rangeLow.toFixed(2).replace(".", ",")}–{rangeHigh.toFixed(2).replace(".", ",")}
        </strong>
        ) ou essayez un autre modèle sur la page d&apos;accueil.
      </p>
      <Link to="/" className="btn btn-primary" style={{ marginTop: "1rem", display: "inline-flex" }}>
        Nouvelle analyse
      </Link>
    </div>
  );
}
