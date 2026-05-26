export function ConfusionPage() {
  return (
    <>
      <h1 className="page-title">Matrice de confusion</h1>
      <p className="page-subtitle">Matrice globale du projet — évaluation sur l&apos;ensemble d&apos;entraînement (indépendante de l&apos;analyse d&apos;un extrait).</p>
      <div className="card empty-state" style={{ padding: "3rem 2rem" }}>
        <p style={{ margin: 0, fontSize: "1.1rem", color: "var(--text-muted)" }}>En construction</p>
      </div>
    </>
  );
}
