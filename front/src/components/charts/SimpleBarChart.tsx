export function SimpleBarChart({
  items,
  unit = "",
  maxValue,
}: {
  items: { label: string; value: number; color?: string }[];
  unit?: string;
  maxValue?: number;
}) {
  const max = maxValue ?? Math.max(...items.map((i) => i.value), 1);
  const w = 600;
  const h = 40 + items.length * 34;
  const pad = { left: 120, top: 16, right: 60 };
  const barH = 22;
  const chartW = w - pad.left - pad.right;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="chart-svg">
      {items.map((item, i) => {
        const y = pad.top + i * (barH + 8);
        const bw = (item.value / max) * chartW;
        return (
          <g key={item.label}>
            <text x={pad.left - 8} y={y + barH / 2} textAnchor="end" dominantBaseline="middle" fill="var(--text-muted)" fontSize="11">
              {item.label.length > 16 ? item.label.slice(0, 14) + "…" : item.label}
            </text>
            <rect x={pad.left} y={y} width={Math.max(bw, 2)} height={barH} fill={item.color ?? "var(--primary)"} rx={4} />
            <text x={pad.left + bw + 6} y={y + barH / 2} dominantBaseline="middle" fontSize="11" fontFamily="var(--mono)">
              {item.value.toFixed(1)}
              {unit}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
