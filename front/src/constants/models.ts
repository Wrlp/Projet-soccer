/** Infos UI par modèle (seuils, messages) — les métadonnées serveur viennent de GET /api/models. */
export interface ModelUiHints {
  thresholdHint: string;
  emptyScoreHint?: string;
  suggestedThresholdRange: [number, number];
  highlightWindowSec: number;
}

export interface ModelOption {
  id: string;
  name: string;
  description: string;
  defaultThreshold: number;
  available?: boolean;
  windowSeconds?: number;
  thresholdHint: string;
  emptyScoreHint?: string;
  suggestedThresholdRange: [number, number];
  highlightWindowSec: number;
}

const MODEL_UI_HINTS: Record<string, ModelUiHints> = {
  videomae: {
    thresholdHint: "VideoMAE : scores souvent plus élevés. Défaut 35 %.",
    suggestedThresholdRange: [0.2, 0.3],
    highlightWindowSec: 1,
  },
  slowfast: {
    thresholdHint: "SlowFast : scores souvent entre 15–35 %. Défaut 18 %.",
    emptyScoreHint: "SlowFast produit souvent des scores entre 15 % et 35 % sur des extraits hors SoccerNet.",
    suggestedThresholdRange: [0.15, 0.25],
    highlightWindowSec: 2,
  },
};

/** Fallback hors-ligne si l'API n'est pas joignable. */
export const FALLBACK_MODEL_OPTIONS: ModelOption[] = [
  {
    id: "videomae",
    name: "VideoMAE",
    description: "VideoMAE SoccerNet — fenêtres ~1 s @ 16 fps",
    defaultThreshold: 0.35,
    windowSeconds: 1,
    ...MODEL_UI_HINTS.videomae,
  },
  {
    id: "slowfast",
    name: "SlowFast",
    description: "SlowFast — fenêtres ~4 s @ 16 fps",
    defaultThreshold: 0.18,
    windowSeconds: 4,
    ...MODEL_UI_HINTS.slowfast,
  },
];

export const DEFAULT_MODEL = "videomae";

export function enrichModelFromApi(apiModel: {
  id: string;
  name: string;
  description: string;
  defaultThreshold: number;
  available?: boolean;
  windowSeconds?: number;
}): ModelOption {
  const hints = MODEL_UI_HINTS[apiModel.id] ?? {
    thresholdHint: `Seuil par défaut : ${(apiModel.defaultThreshold * 100).toFixed(0)} %.`,
    suggestedThresholdRange: [0.15, 0.35] as [number, number],
    highlightWindowSec: apiModel.windowSeconds ?? 1,
  };
  return { ...apiModel, ...hints };
}

export function getModelOption(id: string, options: ModelOption[] = FALLBACK_MODEL_OPTIONS): ModelOption {
  return options.find((m) => m.id === id) ?? options[0];
}

/** Résout un modèle à partir de l'id job ou du nom affiché dans les résultats (ex. "VideoMAE"). */
export function resolveModel(
  key: string | undefined,
  options: ModelOption[] = FALLBACK_MODEL_OPTIONS,
): ModelOption | undefined {
  if (!key) return undefined;
  const lower = key.toLowerCase();
  return (
    options.find((m) => m.id === lower) ??
    options.find((m) => m.name.toLowerCase() === lower) ??
    options.find((m) => lower.includes(m.id) || lower.includes(m.name.toLowerCase()))
  );
}

export function highlightWindowForResult(
  meta: { model?: string; windowSeconds?: number } | undefined,
  options: ModelOption[] = FALLBACK_MODEL_OPTIONS,
): number {
  if (meta?.windowSeconds) return Math.max(0.5, meta.windowSeconds / 2);
  const model = resolveModel(meta?.model, options);
  return model?.highlightWindowSec ?? 1;
}
