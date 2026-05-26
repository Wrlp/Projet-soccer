import { Link, NavLink, Outlet, useParams } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";

const analysisTabs = [
  { to: "timeline", label: "Timeline" },
  { to: "evenements", label: "Événements" },
] as const;

const projectTabs = [
  { to: "/matrice", label: "Matrice" },
  { to: "/entrainement", label: "Entraînement" },
] as const;

export function Layout() {
  const { jobId: routeJobId } = useParams<{ jobId?: string }>();
  const { jobId: ctxJobId } = useAnalysis();
  const jobId = routeJobId ?? ctxJobId;

  return (
    <div className="app-shell">
      <header className="app-header">
        <Link to="/" className="logo">
          <div className="logo-icon">⚽</div>
          <div className="logo-text">
            <strong>SportInsight AI</strong>
            <span>Détection d&apos;événements SoccerNet</span>
          </div>
        </Link>
        <nav className="nav-tabs">
          {jobId &&
            analysisTabs.map(({ to, label }) => (
              <NavLink
                key={to}
                to={`/resultats/${jobId}/${to}`}
                className={({ isActive }) => `nav-tab${isActive ? " active" : ""}`}
              >
                {label}
              </NavLink>
            ))}
          {jobId && <span className="nav-sep" aria-hidden />}
          {projectTabs.map(({ to, label }) => (
            <NavLink key={to} to={to} className={({ isActive }) => `nav-tab nav-tab-project${isActive ? " active" : ""}`}>
              {label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
