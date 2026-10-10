import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { CompassJob, CompassResult } from "../../types";
import { MAP, TIER_NAME, fitToRadius, polar, themeColor, tierOf, tierThresholds, wedgePath } from "../../lib/compass";
import { prefersReducedMotion, useAnimatedNumber } from "./useAnimatedNumber";

interface Props {
  result: CompassResult;
  /** "all" or a theme id. */
  themeFilter: string;
  selectedId: string | null;
  /** A job highlighted from elsewhere on the page (hovering a suggestion card). */
  highlightId: string | null;
  stretch: number;
  onSelect: (jobId: string) => void;
}

interface Tip { job: CompassJob; x: number; y: number }

/**
 * You are at the centre. Distance from the centre is how closely a listing matches your profile; angle groups listings
 * into themes. Dots are jobs you picked, diamonds are suggestions. Marks fly out from the centre when the map mounts.
 */
export default function CompassMap({ result, themeFilter, selectedId, highlightId, stretch, onSelect }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [placed, setPlaced] = useState(prefersReducedMotion());
  const [tip, setTip] = useState<Tip | null>(null);

  useEffect(() => {
    if (placed) return;
    const t = setTimeout(() => setPlaced(true), 60);
    return () => clearTimeout(t);
  }, [placed]);

  const thresholds = tierThresholds(stretch);
  const ringBest = useAnimatedNumber(fitToRadius(thresholds.best));
  const ringReach = useAnimatedNumber(fitToRadius(thresholds.reach));
  const ringStretch = useAnimatedNumber(fitToRadius(Math.max(18, thresholds.reach - 28)));

  const strongest = result.themes.find((t) => t.id === result.strongestThemeId) ?? result.themes[0];
  const matches = (j: CompassJob) => themeFilter === "all" || j.themeId === themeFilter;

  const showTip = (job: CompassJob, x: number, y: number) => setTip({ job, x, y });
  const onPointer = (job: CompassJob) => (e: PointerEvent) => {
    const box = wrap.current?.getBoundingClientRect();
    if (box) showTip(job, e.clientX - box.left + 14, e.clientY - box.top + 14);
  };
  const onFocusDot = (job: CompassJob) => (e: { currentTarget: Element }) => {
    const box = wrap.current?.getBoundingClientRect();
    const r = e.currentTarget.getBoundingClientRect();
    if (box) showTip(job, r.left - box.left + 20, r.top - box.top + 20);
  };
  const onKey = (job: CompassJob) => (e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect(job.job.id);
    }
  };

  return (
    <div className="cp-mapwrap" ref={wrap}>
      <svg
        viewBox={`0 0 ${MAP.size} ${MAP.size}`}
        className="cp-map"
        role="img"
        aria-label="Compass map of the jobs you picked and suggested jobs. Use the table view for the same information as a list."
      >
        {result.themes.map((t) => (
          <path key={t.id} d={wedgePath(t.angle, t.halfWidth, MAP.outer)} style={{ fill: themeColor(t.id), opacity: 0.07 }} />
        ))}
        {[ringBest, ringReach, ringStretch].map((r, i) => (
          <circle key={i} cx={MAP.cx} cy={MAP.cy} r={r} fill="none" style={{ stroke: "var(--cp-grid)", strokeWidth: 1 }} />
        ))}
        <circle cx={MAP.cx} cy={MAP.cy} r={MAP.outer} fill="none" style={{ stroke: "var(--cp-axis)", strokeWidth: 1 }} />
        {[0, 90, 180, 270].map((d) => {
          const [x0, y0] = polar(d, MAP.outer);
          const [x1, y1] = polar(d, MAP.outer + 12);
          return <line key={d} x1={x0} y1={y0} x2={x1} y2={y1} style={{ stroke: "var(--cp-axis)", strokeWidth: 2 }} />;
        })}
        {([["N", 0], ["E", 90], ["S", 180], ["W", 270]] as const).map(([label, d]) => {
          const [x, y] = polar(d, MAP.outer + 26);
          return <text key={label} x={x} y={y + 4} textAnchor="middle" className="cp-map__small">{label}</text>;
        })}
        {([["best", ringBest], ["reach", ringReach], ["stretch", ringStretch]] as const).map(([tier, r]) => (
          <text key={tier} x={MAP.cx + 6} y={MAP.cy - r - 6} className="cp-map__small">{TIER_NAME[tier]}</text>
        ))}
        {result.themes.map((t) => {
          const [x, y] = polar(t.angle, 252);
          return (
            <g key={t.id} className="cp-themelabel" style={{ opacity: placed ? 1 : 0 }}>
              <circle cx={x - 4} cy={y - 14} r={4} style={{ fill: themeColor(t.id) }} />
              <text x={x + 4} y={y - 10} className="cp-map__theme">{t.name.length > 26 ? `${t.name.slice(0, 25)}…` : t.name}</text>
            </g>
          );
        })}

        {strongest && (
          <g
            style={{
              transformOrigin: `${MAP.cx}px ${MAP.cy}px`,
              transformBox: "view-box",
              transform: `rotate(${placed ? strongest.angle : 0}deg)`,
              opacity: placed ? 1 : 0,
              transition: "transform 1.2s cubic-bezier(.2,1.2,.3,1) 0.9s, opacity .6s .9s",
            }}
          >
            <line x1={MAP.cx} y1={MAP.cy - 18} x2={MAP.cx} y2={MAP.cy - 150} strokeWidth={3} strokeLinecap="round" style={{ stroke: "var(--accent)" }} />
            <path d={`M${MAP.cx} ${MAP.cy - 168} l-9 20 h18 z`} style={{ fill: "var(--accent)" }} />
          </g>
        )}

        <circle cx={MAP.cx} cy={MAP.cy} r={22} style={{ fill: "var(--surface)", stroke: "var(--accent)", strokeWidth: 3 }} />
        <text x={MAP.cx} y={MAP.cy + 4} textAnchor="middle" className="cp-map__you">You</text>

        {result.jobs.map((j, i) => {
          const theme = result.themes.find((t) => t.id === j.themeId) ?? result.themes[0];
          const [x, y] = polar(theme.angle + j.offset, fitToRadius(j.fit));
          const visible = matches(j);
          const selected = selectedId === j.job.id;
          const color = themeColor(j.themeId);
          const dim = highlightId !== null && highlightId !== j.job.id;
          const lit = highlightId === j.job.id;
          return (
            <g
              key={j.job.id}
              className={`cp-dot${dim ? " is-dim" : ""}${lit ? " is-lit" : ""}`}
              tabIndex={visible ? 0 : -1}
              role="button"
              aria-label={`${j.job.title} at ${j.job.company.name}, ${theme.name}, fit ${j.fit} percent, ${TIER_NAME[tierOf(j.fit, stretch)]}${j.pick ? "" : ", suggested"}`}
              style={{
                transform: placed ? `translate(${x}px, ${y}px)` : `translate(${MAP.cx}px, ${MAP.cy}px)`,
                opacity: placed ? (visible ? 1 : 0.12) : 0,
                pointerEvents: visible ? "auto" : "none",
                transitionDelay: placed && !prefersReducedMotion() ? `${150 + i * 70}ms` : "0ms",
              }}
              onClick={() => onSelect(j.job.id)}
              onKeyDown={onKey(j)}
              onPointerMove={onPointer(j)}
              onPointerLeave={() => setTip(null)}
              onFocus={onFocusDot(j)}
              onBlur={() => setTip(null)}
            >
              <circle r={16} fill="transparent" />
              {j.pick ? (
                <circle className="cp-mark" r={8} strokeWidth={selected ? 3 : 2} style={{ fill: color, stroke: selected ? "var(--text)" : "var(--surface)" }} />
              ) : (
                <>
                  <circle className="cp-halo" r={9} style={{ fill: color, opacity: 0.35 }} />
                  <rect className="cp-mark" x={-7} y={-7} width={14} height={14} rx={2} transform="rotate(45)" strokeWidth={selected ? 3 : 2} style={{ fill: color, stroke: selected ? "var(--text)" : "var(--surface)" }} />
                </>
              )}
            </g>
          );
        })}
      </svg>

      {tip && (
        <div className="cp-tip" role="status" style={{ left: Math.min(tip.x, (wrap.current?.clientWidth ?? 400) - 250), top: tip.y }}>
          <b>{tip.job.job.title}</b>
          <i>{tip.job.job.company.name}{tip.job.pick ? "" : " · suggested"}</i>
          <div>{result.themes.find((t) => t.id === tip.job.themeId)?.name} · {TIER_NAME[tierOf(tip.job.fit, stretch)]} · fit {tip.job.fit}%</div>
        </div>
      )}

      <div className="cp-legend">
        {result.themes.map((t) => (
          <span key={t.id}><i className="cp-key" style={{ background: themeColor(t.id) }} />{t.name}</span>
        ))}
        <span><i className="cp-key" style={{ background: "var(--text-muted)" }} />A job you picked</span>
        <span><i className="cp-key cp-key--diamond" style={{ background: "var(--text-muted)" }} />A suggestion</span>
        <span><i className="cp-key" style={{ background: "var(--accent)" }} />Your strongest direction</span>
      </div>
    </div>
  );
}
