import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import { pollUntilComplete, uploadAndAnalyze } from "../api/client";
import { useAnalysis } from "../context/AnalysisContext";

export function UploadPage() {
  const navigate = useNavigate();
  const { loadResults, clear } = useAnalysis();
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.18);
  const [contextFrames, setContextFrames] = useState(0);
  const [half, setHalf] = useState<"auto" | "1" | "2">("auto");

  const submit = async () => {
    if (!file) return setError("Choisissez une vidéo");
    setBusy(true);
    setError(null);
    clear();
    try {
      const { jobId } = await uploadAndAnalyze(file, {
        threshold,
        contextFrames,
        half: half === "auto" ? "auto" : (Number(half) as 1 | 2),
      });
      await pollUntilComplete(jobId, setProgress);
      await loadResults(jobId);
      navigate(`/resultats/${jobId}/timeline`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
    } finally {
      setBusy(false);
    }
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  }, []);

  return (
    <>
      <h1 className="page-title">Analyser un extrait vidéo</h1>
      <p className="page-subtitle">
        Analyse par <strong>SlowFast</strong> (checkpoints/best.pth) — fenêtres ~4 s, temps relatif au clip (t=0).
        MP4, MKV, WebM.
      </p>
      <div className="card-grid card-grid-2" style={{ alignItems: "start" }}>
        <div className="card">
          <div
            className={`upload-zone${file ? " has-file" : ""}${drag ? " dragover" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
            onClick={() => document.getElementById("v")?.click()}
          >
            <input id="v" type="file" accept="video/*,.mkv,.mp4,.webm" hidden onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            <div className="upload-icon">🎬</div>
            {file ? <strong>{file.name}</strong> : <strong>Glissez ou cliquez</strong>}
          </div>
          {error && <p style={{ color: "var(--danger)", marginTop: "1rem" }}>{error}</p>}
          <button className="btn btn-primary" style={{ width: "100%", marginTop: "1.5rem" }} disabled={!file || busy} onClick={submit}>
            {busy ? `Analyse… ${progress}%` : "Lancer l'analyse"}
          </button>
        </div>
        <div className="card">
          <h3 style={{ marginBottom: "1rem" }}>Paramètres</h3>
          <div className="form-group">
            <label>
              Seuil — <span className="range-value">{threshold.toFixed(2)}</span>
            </label>
            <input type="range" min={0.05} max={0.9} step={0.05} value={threshold} onChange={(e) => setThreshold(+e.target.value)} />
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.35rem" }}>
              SlowFast : scores souvent entre 15–35 %. Défaut 18 %.
            </p>
          </div>
          <div className="form-group">
            <label>
              Contexte frames — <span className="range-value">{contextFrames}</span>
            </label>
            <input type="range" min={0} max={10} step={1} value={contextFrames} onChange={(e) => setContextFrames(+e.target.value)} />
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              Modèle : SlowFast (fenêtres ~4 s). Paramètre conservé pour compatibilité.
            </p>
          </div>
          <div className="form-group">
            <label>Type de vidéo</label>
            <select value={half} onChange={(e) => setHalf(e.target.value as "auto" | "1" | "2")}>
              <option value="auto">Auto — extrait ou match détecté</option>
              <option value="1">Extrait / 1ère mi-temps</option>
              <option value="2">2ème mi-temps</option>
            </select>
          </div>
        </div>
      </div>
      {busy && (
        <div className="processing-overlay">
          <div className="card processing-card">
            <div className="spinner" />
            <p>Analyse en cours…</p>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
