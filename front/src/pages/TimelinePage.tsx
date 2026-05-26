import { EmptyPredictionsHint } from "../components/EmptyPredictionsHint";
import { RequireResults } from "../components/RequireResults";
import { VideoTimelinePlayer } from "../components/VideoTimelinePlayer";
import { useAnalysis } from "../context/AnalysisContext";

function TimelineContent() {
  const { result, jobId } = useAnalysis();
  if (!result || !jobId) return null;

  return (
    <>
      <h1 className="page-title">Timeline</h1>
      <p className="page-subtitle">
        {result.job.videoName}
        {result.meta && (
          <>
            {" "}
            — {result.meta.segmentLabel ?? "Extrait"} ({result.meta.durationMinutes} min
            {result.meta.isExcerpt ? ", temps relatif au clip" : ""})
          </>
        )}
      </p>
      {result.predictions.length === 0 && <EmptyPredictionsHint result={result} />}
      <VideoTimelinePlayer jobId={jobId} result={result} />
    </>
  );
}

export function TimelinePage() {
  return (
    <RequireResults>
      <TimelineContent />
    </RequireResults>
  );
}
