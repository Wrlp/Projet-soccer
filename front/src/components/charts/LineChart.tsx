export function LineChart({
  series,
  xLabels,
  yMax,
}: {
  series: { label: string; data: number[]; color: string; dashed?: boolean }[];
  xLabels: number[];
  yMax?: number;
}) {
  const pad = { top: 24, right: 24, bottom: 36, left: 48 };
  const w = 560;
  const h = 220;
  const cw = w - pad.left - pad.right;
  const ch = h - pad.top - pad.bottom;
  const maxY = yMax ?? Math.max(...series.flatMap((s) => s.data), 0.01);
  const n = xLabels.length;
  const toX = (i: number) => pad.left + (i / Math.max(n - 1, 1)) * cw;
  const toY = (v: number) => pad.top + ch - (v / maxY) * ch;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="chart-svg">
      {series.map((s) => (
        <polyline
          key={s.label}
          points={s.data.map((v, i) => `${toX(i)},${toY(v)}`).join(" ")}
          fill="none"
          stroke={s.color}
          strokeWidth={2}
          strokeDasharray={s.dashed ? "6 4" : undefined}
        />
      ))}
      {xLabels.filter((_, i) => i % Math.ceil(n / 6) === 0 || i === n - 1).map((x) => (
        <text key={x} x={toX(xLabels.indexOf(x))} y={h - 8} textAnchor="middle" fill="var(--text-muted)" fontSize="10">
          {x.toFixed(1)}
        </text>
      ))}
    </svg>
  );
}
