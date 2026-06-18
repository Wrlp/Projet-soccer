import { Navigate, Route, Routes, useParams } from "react-router-dom";
import { AnalysisProvider } from "./context/AnalysisContext";
import { Layout } from "./components/Layout";
import { SyncJobResults } from "./components/SyncJobResults";
import { UploadPage } from "./pages/UploadPage";
import { TimelinePage } from "./pages/TimelinePage";
import { EventsPage } from "./pages/EventsPage";
import { FiguresPage } from "./pages/FiguresPage";

function RedirectTimeline() {
  const { jobId } = useParams();
  return <Navigate to={`/resultats/${jobId}/timeline`} replace />;
}

export default function App() {
  return (
    <AnalysisProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<UploadPage />} />
          <Route path="/figures" element={<FiguresPage />} />
          <Route path="/resultats/:jobId/matrice" element={<Navigate to="/figures" replace />} />
          <Route path="/resultats/:jobId/entrainement" element={<Navigate to="/figures" replace />} />
          <Route path="/resultats/:jobId/metriques" element={<RedirectTimeline />} />
          <Route path="/resultats/:jobId/classes" element={<RedirectTimeline />} />
          <Route path="/resultats/:jobId" element={<RedirectTimeline />} />
          <Route
            path="/resultats/:jobId/*"
            element={
              <SyncJobResults>
                <Routes>
                  <Route index element={<RedirectTimeline />} />
                  <Route path="timeline" element={<TimelinePage />} />
                  <Route path="evenements" element={<EventsPage />} />
                </Routes>
              </SyncJobResults>
            }
          />
        </Route>
      </Routes>
    </AnalysisProvider>
  );
}
