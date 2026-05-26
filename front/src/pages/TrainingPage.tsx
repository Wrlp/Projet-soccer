export function TrainingPage() {
  return (
    <>
      <h1 className="page-title">Entraînement</h1>
      <p className="page-subtitle">Courbes et métriques d&apos;entraînement globales du projet (indépendantes de l&apos;analyse d&apos;un extrait).</p>
      <div className="card empty-state" style={{ padding: "3rem 2rem" }}>
        <p style={{ margin: 0, fontSize: "1.1rem", color: "var(--text-muted)" }}>En construction</p>
      </div>
    </>
  );
}
