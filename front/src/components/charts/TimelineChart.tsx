import { useCallback, useRef } from "react";
import { eventColor, eventEmoji, timelineMinute, timelineTicks } from "../../constants/events";
import type { DetectedEvent, GroundTruthEvent } from "../../types";
import { isEventActive } from "../../utils/eventHighlight";

const EMOJI_SIZE = 20;
const EMOJI_SIZE_ACTIVE = 26;

export function TimelineChart({
  predictions,
  groundTruth,
  durationMinutes = 90,
  isExcerpt = false,
  playheadSeconds = 0,
  highlightWindowSec,
  onSeek,
}: {
  predictions: DetectedEvent[];
  groundTruth?: GroundTruthEvent[];
  durationMinutes?: number;
  isExcerpt?: boolean;
  playheadSeconds?: number;
  highlightWindowSec?: number;
  onSeek?: (seconds: number) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const maxMin = Math.max(durationMinutes, 0.5);
  const pad = { top: 44, right: 20, bottom: 32, left: 48 };
  const w = 900;
  const rows = groundTruth?.length ? 2 : 1;
  const h = 88 + rows * 60;
  const chartW = w - pad.left - pad.right;
  const toX = (m: number) => pad.left + (m / maxMin) * chartW;
  const ticks = timelineTicks(maxMin);
  const showHalftime = !isExcerpt && maxMin > 50;
  const playheadMin = Math.min(Math.max(0, playheadSeconds / 60), maxMin);
  const playheadX = toX(playheadMin);

  const seekFromClientX = useCallback(
    (clientX: number) => {
      if (!onSeek || !svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      const scale = w / rect.width;
      const xSvg = (clientX - rect.left) * scale;
      const minute = ((xSvg - pad.left) / chartW) * maxMin;
      const clamped = Math.max(0, Math.min(maxMin, minute));
      onSeek(clamped * 60);
    },
    [onSeek, chartW, maxMin, pad.left]
  );

  const handlePointer = (e: React.MouseEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    seekFromClientX(e.clientX);
  };

  const row = (events: { timestamp: number; half: 1 | 2; label: string }[], y: number, title: string) => (
    <g key={title}>
      <text x={pad.left} y={y - 14} fill="var(--text-muted)" fontSize="11" fontWeight="600">
        {title}
      </text>
      <line x1={pad.left} y1={y + 10} x2={pad.left + chartW} y2={y + 10} stroke="var(--border)" />
      {events.map((ev, i) => {
        const cx = toX(timelineMinute(ev.timestamp, ev.half, isExcerpt, maxMin));
        const cy = y + 14;
        const emoji = eventEmoji(ev.label);
        const active = isEventActive(ev.timestamp, playheadSeconds, highlightWindowSec);
        const color = eventColor(ev.label);
        const seekTo = () => onSeek?.(ev.timestamp);
        return (
          <g
            key={i}
            style={{ cursor: onSeek ? "pointer" : undefined }}
            onClick={(e) => {
              e.stopPropagation();
              seekTo();
            }}
          >
            <title>{`${ev.label} — ${ev.timestamp}s`}</title>
            {active && (
              <circle cx={cx} cy={cy} r={18} fill={color} fillOpacity={0.35} stroke={color} strokeWidth={2} />
            )}
            <text
              x={cx}
              y={cy}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize={active ? EMOJI_SIZE_ACTIVE : EMOJI_SIZE}
              className={`timeline-emoji${active ? " timeline-emoji--active" : ""}`}
            >
              {emoji}
            </text>
          </g>
        );
      })}
    </g>
  );

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${w} ${h}`}
      className={`chart-svg${onSeek ? " chart-svg-interactive" : ""}`}
      onClick={handlePointer}
      role={onSeek ? "slider" : undefined}
      aria-label={onSeek ? "Timeline cliquable" : undefined}
    >
      {showHalftime && (
        <>
          <line x1={toX(45)} y1={pad.top - 8} x2={toX(45)} y2={h - pad.bottom} stroke="var(--text-muted)" strokeDasharray="6 4" />
          <text x={toX(45)} y={pad.top - 14} textAnchor="middle" fill="var(--text-muted)" fontSize="10">
            Mi-temps
          </text>
        </>
      )}
      {ticks.map((m) => (
        <text key={m} x={toX(m)} y={h - 8} textAnchor="middle" fill="var(--text-muted)" fontSize="10">
          {isExcerpt ? (m < 1 ? `${Math.round(m * 60)}s` : `${m.toFixed(0)} min`) : `${m}'`}
        </text>
      ))}
      {row(predictions, pad.top + 24, "Prédictions")}
      {groundTruth?.length ? row(groundTruth, pad.top + 84, "Vérité terrain") : null}
      <line
        x1={playheadX}
        y1={pad.top - 4}
        x2={playheadX}
        y2={h - pad.bottom}
        stroke="#f43f5e"
        strokeWidth={2.5}
        pointerEvents="none"
      />
      <polygon
        points={`${playheadX},${pad.top - 4} ${playheadX - 7},${pad.top - 16} ${playheadX + 7},${pad.top - 16}`}
        fill="#f43f5e"
        pointerEvents="none"
      />
    </svg>
  );
}
