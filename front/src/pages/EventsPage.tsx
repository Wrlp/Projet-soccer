import { EmptyPredictionsHint } from "../components/EmptyPredictionsHint";
import { EventChip } from "../components/EventChip";
import { RequireResults } from "../components/RequireResults";
import { formatTime } from "../constants/events";
import { useAnalysis } from "../context/AnalysisContext";

function EventsContent() {
  const { result } = useAnalysis();
  if (!result) return null;

  const isExcerpt = result.meta?.isExcerpt ?? true;
  const sorted = [...result.predictions].sort((a, b) => a.timestamp - b.timestamp);

  return (
    <>
      <h1 className="page-title">Événements détectés</h1>
      {sorted.length === 0 && <EmptyPredictionsHint result={result} />}
      {result.meta?.usedFallback && sorted.length > 0 && (
        <p style={{ color: "var(--text-muted)", marginBottom: "1rem", fontSize: "0.9rem" }}>
          {result.meta.detectionHint ?? "Résultats issus du mode secours (meilleures fenêtres)."}
        </p>
      )}
      <div className="card table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Type</th>
              <th>{isExcerpt ? "Temps (dans le clip)" : "Temps"}</th>
              <th>Confiance</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((ev, i) => (
              <tr key={i}>
                <td>
                  <EventChip label={ev.label} confidence={ev.confidence} />
                </td>
                <td style={{ fontFamily: "var(--mono)" }}>{formatTime(ev.timestamp, ev.half, isExcerpt)}</td>
                <td>{(ev.confidence * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export function EventsPage() {
  return (
    <RequireResults>
      <EventsContent />
    </RequireResults>
  );
}
