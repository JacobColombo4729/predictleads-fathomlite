interface Props {
  values: number[];
  weekEnds: string[];
  /** Colour class suffix, so the pattern sets the bar colour. */
  tone: string;
  width?: number;
  height?: number;
}

/** Weekly new openings as thin bars. The title on each bar gives the exact count. */
export default function Sparkline({ values, weekEnds, tone, width = 156, height = 36 }: Props) {
  const max = Math.max(1, ...values);
  const gap = 2;
  const bar = Math.max(2, (width - gap * (values.length - 1)) / Math.max(1, values.length));
  return (
    <svg className={`sg-spark sg-spark--${tone}`} width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img"
      aria-label={`New openings per week over the last ${values.length} weeks; most in one week: ${max}`}>
      <line x1="0" y1={height - 0.5} x2={width} y2={height - 0.5} className="sg-spark__base" />
      {values.map((v, i) => {
        const h = v === 0 ? 0 : Math.max(2, (v / max) * (height - 3));
        return (
          <rect key={i} x={i * (bar + gap)} y={height - 1 - h} width={bar} height={h} rx="1">
            <title>{`Week ending ${weekEnds[i] ?? ""}: ${v} new`}</title>
          </rect>
        );
      })}
    </svg>
  );
}
