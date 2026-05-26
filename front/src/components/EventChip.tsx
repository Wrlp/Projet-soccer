import { eventColor } from "../constants/events";

interface EventChipProps {
  label: string;
  confidence?: number;
  time?: string;
}

export function EventChip({ label, confidence, time }: EventChipProps) {
  const color = eventColor(label);
  return (
    <span className="event-chip" style={{ borderLeftColor: color }}>
      <span style={{ width: 8, height: 8, borderRadius: "50%", background: color }} />
      {label}
      {time && <span style={{ color: "var(--text-muted)", marginLeft: 4 }}>{time}</span>}
      {confidence !== undefined && (
        <span style={{ color: "var(--text-muted)", fontFamily: "var(--mono)", fontSize: "0.75rem" }}>
          {(confidence * 100).toFixed(0)}%
        </span>
      )}
    </span>
  );
}
