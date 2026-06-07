import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchModels, pollUntilComplete, uploadAndAnalyze } from "../api/client";
import {
  DEFAULT_MODEL,
  FALLBACK_MODEL_OPTIONS,
  getModelOption,
  type ModelOption,
} from "../constants/models";
import { useAnalysis } from "../context/AnalysisContext";

export function UploadPage() {
  const navigate = useNavigate();
  const { loadResults, clear } = useAnalysis();
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>(FALLBACK_MODEL_OPTIONS);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [threshold, setThreshold] = useState(getModelOption(DEFAULT_MODEL).defaultThreshold);
  const [strideSec, setStrideSec] = useState(1);
  const [half, setHalf] = useState<"auto" | "1" | "2">("auto");

  useEffect(() => {
    fetchModels()
      .then(({ defaultModel, models }) => {
        setModelOptions(models);
        setModel(defaultModel);
        setThreshold(getModelOption(defaultModel, models).defaultThreshold);
      })
      .catch(() => {
        setModelOptions(FALLBACK_MODEL_OPTIONS);
      });
  }, []);

  const modelInfo = getModelOption(model, modelOptions);

  const onModelChange = (next: string) => {
    setModel(next);
    setThreshold(getModelOption(next, modelOptions).defaultThreshold);
  };

  const submit = async () => {
    if (!file) return setError("Choisissez une vidéo");
    if (modelInfo.available === false) {
      return setError(`Le modèle ${modelInfo.name} n'est pas disponible sur le serveur.`);
    }
    setBusy(true);
    setError(null);
    clear();
    try {
      const { jobId } = await uploadAndAnalyze(file, {
        threshold,
        strideSec,
        half: half === "auto" ? "auto" : (Number(half) as 1 | 2),
        model,
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
        Analyse par <strong>{modelInfo.name}</strong> — {modelInfo.description}. Temps relatif au clip (t=0). MP4, MKV,
        WebM.
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
            <label>Modèle d&apos;analyse</label>
            <select value={model} onChange={(e) => onModelChange(e.target.value)}>
              {modelOptions.map((m) => (
                <option key={m.id} value={m.id} disabled={m.available === false}>
                  {m.name}
                  {m.available === false ? " (indisponible)" : ""}
                </option>
              ))}
            </select>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.35rem" }}>{modelInfo.description}</p>
          </div>
          <div className="form-group">
            <label>
              Seuil — <span className="range-value">{threshold.toFixed(2)}</span>
            </label>
            <input type="range" min={0.05} max={0.9} step={0.05} value={threshold} onChange={(e) => setThreshold(+e.target.value)} />
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.35rem" }}>{modelInfo.thresholdHint}</p>
          </div>
          <div className="form-group">
            <label>
              Pas d&apos;analyse — <span className="range-value">{strideSec.toFixed(1)} s</span>
            </label>
            <input type="range" min={1} max={5} step={0.5} value={strideSec} onChange={(e) => setStrideSec(+e.target.value)} />
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.35rem" }}>
              Intervalle entre fenêtres (1 s = précis, 2–3 s = plus rapide).
            </p>
          </div>
          <div className="form-group">
            <label>Type de vidéo</label>
            <select value={half} onChange={(e) => setHalf(e.target.value as "auto" | "1" | "2")}>
              <option value="auto">Auto — extrait ou match détecté</option>
              <option value="1">Extrait / 1ère mi-temps</option>
              <option value="2">2ème mi-temps (match long uniquement)</option>
            </select>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.35rem" }}>
              Sur un match complet (&gt;42 min), seule la mi-temps choisie est analysée.
            </p>
          </div>
        </div>
      </div>
      {busy && (
        <div className="processing-overlay">
          <div className="card processing-card">
            <div className="spinner" />
            <p>Analyse en cours avec {modelInfo.name}…</p>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
