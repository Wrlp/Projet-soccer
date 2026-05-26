import type { DetectedEvent } from "../types";

/** Fenêtre SlowFast ≈ 4 s — surbrillance autour du centre de détection */
export const EVENT_HIGHLIGHT_WINDOW_SEC = 1;

export function isEventActive(eventTimestamp: number, playheadSec: number, windowSec = EVENT_HIGHLIGHT_WINDOW_SEC): boolean {
  return Math.abs(playheadSec - eventTimestamp) <= windowSec;
}

export function getActiveEventIndices(events: DetectedEvent[], playheadSec: number, windowSec = EVENT_HIGHLIGHT_WINDOW_SEC): number[] {
  return events.map((ev, i) => (isEventActive(ev.timestamp, playheadSec, windowSec) ? i : -1)).filter((i) => i >= 0);
}
