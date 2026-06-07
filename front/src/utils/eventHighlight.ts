import { highlightWindowForResult } from "../constants/models";
import type { AnalysisResult, DetectedEvent } from "../types";

export const DEFAULT_HIGHLIGHT_WINDOW_SEC = 1;

export function isEventActive(
  eventTimestamp: number,
  playheadSec: number,
  windowSec = DEFAULT_HIGHLIGHT_WINDOW_SEC,
  timeOffsetSec = 0
): boolean {
  return Math.abs(playheadSec - (eventTimestamp + timeOffsetSec)) <= windowSec;
}

export function getActiveEventIndices(
  events: DetectedEvent[],
  playheadSec: number,
  windowSec = DEFAULT_HIGHLIGHT_WINDOW_SEC,
  timeOffsetSec = 0
): number[] {
  return events
    .map((ev, i) => (isEventActive(ev.timestamp, playheadSec, windowSec, timeOffsetSec) ? i : -1))
    .filter((i) => i >= 0);
}

export function highlightWindowFromResult(result: AnalysisResult | null | undefined): number {
  return highlightWindowForResult(result?.meta);
}
