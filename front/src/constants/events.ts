export const EVENT_COLORS: Record<string, string> = {
  Goal: "#eab308",
  "Yellow card": "#facc15",
  "Red card": "#ef4444",
  Corner: "#3b82f6",
  Substitution: "#22c55e",
  Foul: "#f97316",
  "Shots on target": "#a855f7",
  "Shots off target": "#c084fc",
  "Direct free-kick": "#06b6d4",
  "Indirect free-kick": "#67e8f9",
  Offside: "#ec4899",
  Clearance: "#9ca3af",
  "Throw-in": "#d1d5db",
  "Ball out of play": "#6b7280",
  "Kick-off": "#84cc16",
  Penalty: "#dc2626",
  "Yellow->red card": "#ea580c",
};

export function eventColor(label: string): string {
  return EVENT_COLORS[label] ?? "#64748b";
}

/** Emoji affiché sur la timeline (label affiché ou classKey API) */
export const EVENT_EMOJI: Record<string, string> = {
  Goal: "⚽",
  "Yellow card": "🟨",
  "Red card": "🟥",
  "Yellow->red card": "🟨🟥",
  Corner: "🚩",
  Substitution: "🔄",
  Foul: "🤕",
  "Shots on target": "🎯",
  "Shots off target": "↗️",
  "Direct free-kick": "🦵",
  "Indirect free-kick": "🦶",
  Offside: "🚫",
  Clearance: "🛡️",
  "Throw-in": "🤾",
  "Ball out of play": "⏸️",
  "Kick-off": "🏁",
  Penalty: "⚠️",
  // classKey (API)
  Ball_out_of_play: "⏸️",
  Yellow_card: "🟨",
  Red_card: "🟥",
  Yellow_red_card: "🟨🟥",
  Shots_on_target: "🎯",
  Shots_off_target: "↗️",
  Direct_free_kick: "🦵",
  Indirect_free_kick: "🦶",
  Kick_off: "🏁",
  Throw_in: "🤾",
};

export function eventEmoji(label: string): string {
  return EVENT_EMOJI[label] ?? "📍";
}

export function formatTime(seconds: number, half: 1 | 2, isExcerpt = false): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const clock = `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  if (isExcerpt) return `t+${clock}`;
  return `${half === 2 ? "2ᵉ" : "1ʳᵉ"} — ${clock}`;
}

/** Position horizontale sur la timeline (minutes) */
export function timelineMinute(seconds: number, half: 1 | 2, isExcerpt: boolean, maxMinutes: number): number {
  if (isExcerpt) return seconds / 60;
  if (maxMinutes <= 50) return seconds / 60;
  const offset = half === 2 ? 45 : 0;
  return offset + seconds / 60;
}

export function timelineTicks(maxMinutes: number): number[] {
  if (maxMinutes <= 3) return [0, Math.max(1, Math.ceil(maxMinutes))];
  if (maxMinutes <= 15) {
    const step = maxMinutes <= 5 ? 1 : 5;
    return Array.from({ length: Math.ceil(maxMinutes / step) + 1 }, (_, i) => i * step);
  }
  if (maxMinutes <= 50) return [0, 5, 10, 15, 20, 25, 30, 35, 40, 45].filter((t) => t <= maxMinutes);
  return [0, 15, 30, 45, 60, 75, 90].filter((t) => t <= maxMinutes);
}
