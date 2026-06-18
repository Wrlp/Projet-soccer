import { useEffect, useMemo, useState } from "react";
import { fetchFiguresByModel, getFigureUrl } from "../api/client";
import type { ApiFiguresModel } from "../types";

export function FiguresPage() {
  const [models, setModels] = useState<ApiFiguresModel[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [zoomedFigure, setZoomedFigure] = useState<{ file: string; url: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    fetchFiguresByModel()
      .then((data) => {
        if (!active) return;
        setModels(data.models);
        setSelectedModel((prev) => prev || data.models[0]?.id || "");
        setError("");
      })
      .catch((err: unknown) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Impossible de charger les figures");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const selectedFigures = useMemo(
    () => models.find((model) => model.id === selectedModel)?.figures ?? [],
    [models, selectedModel],
  );

  return (
    <>
      <h1 className="page-title">Figures</h1>
      <p className="page-subtitle">Visualisez les figures générées dans `outputs/figures/` et filtrez par modèle.</p>

      <div className="card" style={{ marginBottom: "1rem" }}>
        <div className="form-group" style={{ marginBottom: 0, maxWidth: 420 }}>
          <label htmlFor="figures-model-select">Modèle</label>
          <select
            id="figures-model-select"
            value={selectedModel}
            onChange={(event) => setSelectedModel(event.target.value)}
            disabled={loading || models.length === 0}
          >
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.id}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && (
        <div className="card empty-state">
          <p>Chargement des figures...</p>
        </div>
      )}

      {!loading && error && (
        <div className="card empty-state">
          <p style={{ color: "var(--danger)" }}>{error}</p>
        </div>
      )}

      {!loading && !error && models.length === 0 && (
        <div className="card empty-state">
          <p>Aucune figure disponible dans `outputs/figures/`.</p>
        </div>
      )}

      {!loading && !error && selectedFigures.length > 0 && (
        <div className="figures-grid">
          {selectedFigures.map((file) => (
            <figure key={file} className="card figure-card">
              <figcaption className="figure-caption">{file}</figcaption>
              <button
                type="button"
                className="figure-zoom-trigger"
                onClick={() => setZoomedFigure({ file, url: getFigureUrl(selectedModel, file) })}
                aria-label={`Agrandir ${file}`}
              >
                <img src={getFigureUrl(selectedModel, file)} alt={file} className="figure-image" loading="lazy" />
              </button>
            </figure>
          ))}
        </div>
      )}

      {!loading && !error && models.length > 0 && selectedFigures.length === 0 && (
        <div className="card empty-state">
          <p>Aucune figure pour ce modèle.</p>
        </div>
      )}

      {zoomedFigure && (
        <div className="figure-modal-backdrop" role="presentation" onClick={() => setZoomedFigure(null)}>
          <div className="figure-modal-content" role="dialog" aria-modal="true" aria-label={zoomedFigure.file} onClick={(e) => e.stopPropagation()}>
            <div className="figure-modal-header">
              <strong>{zoomedFigure.file}</strong>
              <button type="button" className="btn btn-secondary" onClick={() => setZoomedFigure(null)}>
                Fermer
              </button>
            </div>
            <img src={zoomedFigure.url} alt={zoomedFigure.file} className="figure-modal-image" />
          </div>
        </div>
      )}
    </>
  );
}
