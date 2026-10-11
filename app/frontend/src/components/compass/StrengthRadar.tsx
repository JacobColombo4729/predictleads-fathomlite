import { useId, useState } from "react";
import { CREDENTIALS_AREA, type CompassAreaScore, type CompassJob } from "../../types";
import { wrapLabel } from "../../lib/compass";

interface Props {
  job: CompassJob;
}

const SIZE = { w: 340, h: 260, cx: 170, cy: 130, r: 82 };
// Coverage thresholds the backend uses for Partial and Supported (backend/compass.py PARTIAL_AT, SUPPORTED_AT).
const RINGS = [{ at: 40, label: "Partial" }, { at: 70, label: "Supported" }, { at: 100, label: "" }];

function point(i: number, n: number, value: number): [number, number] {
  const a = (i / n) * 2 * Math.PI;
  const r = (value / 100) * SIZE.r;
  return [SIZE.cx + r * Math.sin(a), SIZE.cy - r * Math.cos(a)];
}

function polygon(values: number[]): string {
  return values.map((v, i) => point(i, values.length, v).map((c) => c.toFixed(1)).join(",")).join(" ");
}

/**
 * Strengths radar for one job. One spoke per skill area: the grey shape is how much the listing asks for each area,
 * the blue shape how closely the profile's own lines match it. Credentials & eligibility are never scored, so they
 * get no spoke. Hover a spoke for its numbers.
 */
export default function StrengthRadar({ job }: Props) {
  const [hover, setHover] = useState<string | null>(null);
  const gradient = useId();
  const spokes = job.areas.filter((a) => a.id !== CREDENTIALS_AREA);
  const asked = spokes.filter((a) => a.shown !== null);

  if (spokes.length < 3 || asked.length < 2) {
    return <p className="cp-sub cp-gap-sm">This listing has too little requirement text to chart.</p>;
  }
  const n = spokes.length;
  const tip = (a: CompassAreaScore) =>
    a.shown === null ? `${a.name}: this listing doesn’t ask about this`
      : `${a.name}: ${a.lines} requirement ${a.lines === 1 ? "line" : "lines"}; your profile matches ${a.shown}%`;

  return (
    <div className="cp-radar">
      <svg viewBox={`0 0 ${SIZE.w} ${SIZE.h}`} role="img"
        aria-label={`Strengths for this job by skill area. ${asked.map((a) => `${a.name}: your profile matches ${a.shown}%`).join(". ")}.`}>
        <defs>
          <radialGradient id={gradient} cx={SIZE.cx} cy={SIZE.cy} r={SIZE.r} gradientUnits="userSpaceOnUse">
            <stop offset="0%" className="cp-radar__askedstop" stopOpacity={0.06} />
            <stop offset="100%" className="cp-radar__askedstop" stopOpacity={0.26} />
          </radialGradient>
        </defs>
        {RINGS.map((ring) => (
          <polygon key={ring.at} className={`cp-radar__ring${ring.label ? " is-threshold" : ""}`} points={polygon(spokes.map(() => ring.at))} />
        ))}
        {spokes.map((a, i) => {
          const [x, y] = point(i, n, 100);
          return <line key={a.id} className={`cp-radar__spoke${hover === a.id ? " is-focus" : ""}`} x1={SIZE.cx} y1={SIZE.cy} x2={x} y2={y} />;
        })}
        <g className="cp-radar__shapes" key={job.job.id}>
          <polygon className="cp-radar__asked" fill={`url(#${gradient})`} points={polygon(spokes.map((a) => a.asked))} />
          <polygon className="cp-radar__shown" points={polygon(spokes.map((a) => a.shown ?? 0))} />
        </g>
        {RINGS.filter((ring) => ring.label).map((ring) => (
          <text key={ring.at} className="cp-radar__ringlabel" x={SIZE.cx + 4} y={SIZE.cy - (ring.at / 100) * SIZE.r - 3}>{ring.label}</text>
        ))}
        {spokes.map((a, i) => {
          const [x, y] = point(i, n, 118);
          const lines = wrapLabel(a.name, 13);
          const anchor = Math.abs(x - SIZE.cx) < 8 ? "middle" : x > SIZE.cx ? "start" : "end";
          const top = y < SIZE.cy - 8 ? y - (lines.length - 1) * 11 : y > SIZE.cy + 8 ? y + 8 : y - ((lines.length - 1) * 11) / 2 + 4;
          return (
            <text key={a.id} className={`cp-radar__label${a.lines === 0 ? " is-empty" : ""}`} x={x} y={top} textAnchor={anchor}
              onMouseEnter={() => setHover(a.id)} onMouseLeave={() => setHover(null)}>
              <title>{tip(a)}</title>
              {lines.map((line, k) => <tspan key={k} x={x} dy={k ? 11 : 0}>{line}</tspan>)}
            </text>
          );
        })}
      </svg>

      <div className="cp-legend cp-radar__legend">
        <span><i className="cp-key cp-key--asked" />What this job asks for</span>
        <span><i className="cp-key cp-key--shown" />How closely your profile matches</span>
      </div>
    </div>
  );
}
