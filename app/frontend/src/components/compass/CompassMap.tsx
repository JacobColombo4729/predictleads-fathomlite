import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { CompassJob, CompassResult, CompassRole } from "../../types";
import {
  MAP, MAP_PAD, TIER_NAME, facing, fitToRadius, polar, scopeHalfWidth, shortestTurn, themeColor, tierOf,
  tierThresholds, wedgePath, wrapLabel,
} from "../../lib/compass";
import { prefersReducedMotion, useAnimatedNumber } from "./useAnimatedNumber";
import { useDial } from "./useCompassDial";

interface Props {
  result: CompassResult;
  /** "all" or a theme id. */
  themeFilter: string;
  selectedId: string | null;
  /** A job highlighted from elsewhere on the page (hovering a suggestion card). */
  highlightId: string | null;
  stretch: number;
  onSelect: (jobId: string) => void;
  /** Called with the ids of the jobs inside the scope wedge whenever that set changes, best fit first. */
  onScopeChange?: (jobIds: string[]) => void;
}

interface Tip { job: CompassJob; x: number; y: number }
interface Drag {
  pointerId: number;
  /** Pointer heading at the last move, in degrees clockwise from north. */
  last: number;
  moved: boolean;
  /** Map centre in client pixels, measured once when the drag starts (the map doesn't move during a drag). */
  cx: number;
  cy: number;
  el: Element;
  /** Removes the window listeners that end the drag if the release happens somewhere the ring can't see. */
  unlisten: () => void;
}

const CARDINALS = [["N", 0], ["E", 90], ["S", 180], ["W", 270]] as const;
const DEGREE_TICKS = Array.from({ length: 36 }, (_, i) => i * 10);
/** After the dots fly out, the bezel swings round so your strongest direction faces north. */
const SETTLE_DELAY_MS = 900;
const SETTLE_MS = 1400;
/** The in-scope list waits this long after the set stops changing, so it doesn't reshuffle mid-turn. */
const SCOPE_SETTLE_MS = 200;

/** Upright label position for a role spoke at screen heading `a`. */
function labelPlacement(a: number, lineCount: number): { x: number; y: number; anchor: "start" | "middle" | "end" } {
  const [x, y] = polar(a, MAP.outer + 20);
  const sin = Math.sin((a * Math.PI) / 180);
  const cos = Math.cos((a * Math.PI) / 180);
  const lead = cos > 0.3 ? lineCount - 1 : cos < -0.3 ? -0.6 : (lineCount - 1) / 2;
  return { x, y: y + 4 - lead * 13, anchor: sin > 0.3 ? "start" : sin < -0.3 ? "end" : "middle" };
}

/**
 * With more than 4 roles (the backend sends them in ring order, starting at 0 degrees), every other spoke is an
 * in-between role: its label only shows while it is inside the scope wedge; the "Facing" line always names it.
 */
const isMinor = (roles: CompassRole[], index: number) => roles.length > 4 && index % 2 === 1;

const inScopeAt = (angle: number, turn: number, half: number) => Math.abs(shortestTurn(0, angle + turn)) <= half;

const inScopeKey = (roles: CompassRole[], turn: number, half: number) =>
  roles.map((r) => (inScopeAt(r.angle, turn, half) ? "1" : "0")).join("");

/**
 * You are at the centre. Distance from the centre is how closely a listing matches your profile; direction is the
 * listing's role family: up to 4 axes, with the least alike roles facing each other. Colour is the theme. Dots are
 * suggestions, diamonds are the jobs you picked. Marks fly out from the centre when the map mounts.
 *
 * Like a hand compass: the role ring is a bezel you can turn (drag it, use the arrow keys, or click a role to face
 * it) while N/E/S/W and the needle stay still on north. The highlighted scope
 * slice always points north: it opens on the densest, best-fitting direction (which can fall between two roles) and
 * narrows over a tight cluster of good fits, widening over sparse or weak ground.
 *
 * The 4 main spokes are labelled; an in-between spoke's label appears only inside the scope wedge, and every spoke's
 * tick names its role on hover and turns it to north on click. Jobs outside the wedge fade, and the jobs inside it
 * are reported to the page through onScopeChange.
 *
 * Turning never re-renders React: useDial's frame loop writes the moving attributes straight to the SVG, and state
 * changes only when something readable does (which roles or jobs are in scope), a few times a turn rather than every
 * frame. The "Facing" line fades out as soon as the ring moves and fades back in with its new wording once the turn
 * comes to rest (a drag is released or an eased turn arrives); the class toggle happens outside React.
 */
export default function CompassMap({ result, themeFilter, selectedId, highlightId, stretch, onSelect, onScopeChange }: Props) {
  // Clicking a legend entry brings that kind of mark (picked jobs or suggested roles) to the front and keeps it there; the other fades back.
  const [activeGroup, setActiveGroup] = useState<"pick" | "suggested" | null>(null);
  const toggleGroup = (group: "pick" | "suggested") => setActiveGroup((g) => (g === group ? null : group));
  const wrap = useRef<HTMLDivElement>(null);
  const [placed, setPlaced] = useState(prefersReducedMotion());
  const [tip, setTip] = useState<Tip | null>(null);
  const [dragging, setDragging] = useState(false);
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<Drag | null>(null);
  const bezelG = useRef<SVGGElement>(null);
  const hitsG = useRef<SVGGElement>(null);
  const dotsG = useRef<SVGGElement>(null);
  const scopePath = useRef<SVGPathElement>(null);
  const slider = useRef<SVGCircleElement>(null);
  const labels = useRef<(SVGTextElement | null)[]>([]);
  const marks = useRef<(SVGGElement | null)[]>([]);
  const matches = (j: CompassJob) => themeFilter === "all" || j.themeId === themeFilter;
  /** Visible jobs inside the wedge, best fit first. */
  const jobsInScope = (turn: number, half: number) =>
    result.jobs.filter((j) => matches(j) && inScopeAt(j.angle, turn, half)).sort((a, b) => b.fit - a.fit).map((j) => j.job.id);
  const snapshot = (turn: number) => {
    const half = scopeHalfWidth(result.scope, turn);
    return { facing: facing(result.roles, turn), inScope: inScopeKey(result.roles, turn, half), jobs: jobsInScope(turn, half).join(" ") };
  };
  const [view, setView] = useState(() => snapshot(0));
  // what the frame loop last measured; React only hears about it when it changes
  const measured = useRef(view);
  const publish = (next: typeof view) => {
    const m = measured.current;
    if (m.facing === next.facing && m.inScope === next.inScope && m.jobs === next.jobs) return;
    measured.current = next;
    setView(next);
  };
  const lines = result.roles.map((r) => wrapLabel(r.name));

  const [shownFacing, setShownFacing] = useState(view.facing);
  const facingEl = useRef<HTMLParagraphElement>(null);
  /** The turn has come to rest: swap in the new wording while the line is hidden, then let it fade back in. */
  const showFacing = (turn: number) => {
    setShownFacing(facing(result.roles, turn));
    facingEl.current?.classList.remove("is-turning");
  };
  const dial = useDial((turn) => {
    facingEl.current?.classList.add("is-turning");
    const spin = `rotate(${turn} ${MAP.cx} ${MAP.cy})`;
    bezelG.current?.setAttribute("transform", spin);
    hitsG.current?.setAttribute("transform", spin);
    dotsG.current?.setAttribute("transform", spin);
    for (const m of marks.current) m?.setAttribute("transform", `rotate(${-turn})`);
    const half = scopeHalfWidth(result.scope, turn);
    scopePath.current?.setAttribute("d", wedgePath(0, half, MAP.outer));
    result.roles.forEach((r, i) => {
      const el = labels.current[i];
      if (!el) return;
      const at = labelPlacement(r.angle + turn, lines[i].length);
      el.setAttribute("x", String(at.x));
      el.setAttribute("y", String(at.y));
      el.setAttribute("text-anchor", at.anchor);
      el.querySelectorAll("tspan").forEach((t) => t.setAttribute("x", String(at.x)));
    });
    slider.current?.setAttribute("aria-valuenow", String(((Math.round(turn) % 360) + 360) % 360));
    publish(snapshot(turn));
  }, showFacing);
  const turn = dial.turn.current;

  useEffect(() => {
    if (placed) return;
    const t = setTimeout(() => setPlaced(true), 60);
    return () => clearTimeout(t);
  }, [placed]);
  const best = result.scope.heading;
  const { turnTo } = dial;
  useEffect(() => {
    if (prefersReducedMotion()) {
      turnTo(-best); // instant with reduced motion, and still reports rest so the "Facing" line updates
      return;
    }
    const t = setTimeout(() => turnTo(-best, SETTLE_MS), SETTLE_DELAY_MS);
    return () => clearTimeout(t);
  }, [best, turnTo]);

  const thresholds = tierThresholds(stretch);
  const ringBest = useAnimatedNumber(fitToRadius(thresholds.best));
  const ringReach = useAnimatedNumber(fitToRadius(thresholds.reach));
  const ringStretch = useAnimatedNumber(fitToRadius(Math.max(18, thresholds.reach - 28)));

  // the theme filter changes which jobs count; re-measure without waiting for a turn
  useEffect(() => {
    publish(snapshot(dial.turn.current));
  }, [themeFilter]); // eslint-disable-line react-hooks/exhaustive-deps

  const [settledJobs, setSettledJobs] = useState(view.jobs);
  useEffect(() => {
    const t = setTimeout(() => setSettledJobs(view.jobs), SCOPE_SETTLE_MS);
    return () => clearTimeout(t);
  }, [view.jobs]);
  useEffect(() => {
    onScopeChange?.(settledJobs ? settledJobs.split(" ") : []);
  }, [settledJobs, onScopeChange]);
  const scopeJobs = new Set(view.jobs ? view.jobs.split(" ") : []);
  const heading = ((Math.round(turn) % 360) + 360) % 360;

  /** Heading of a client point around the map centre, in degrees clockwise from north. */
  const headingFrom = (cx: number, cy: number, x: number, y: number) => (Math.atan2(x - cx, cy - y) * 180) / Math.PI;

  /**
   * Ends a drag, however it ends: a release on the ring, a release anywhere else on the page, the browser cancelling
   * or dropping the pointer, or a move that arrives with no button held. Returns true when the pointer barely moved,
   * so the press counts as a click.
   */
  const endDrag = (): boolean => {
    const d = drag.current;
    if (!d) return false;
    drag.current = null;
    d.unlisten();
    setDragging(false);
    try {
      if (d.el.hasPointerCapture?.(d.pointerId)) d.el.releasePointerCapture(d.pointerId);
    } catch {
      // the pointer is already gone
    }
    if (d.moved) showFacing(dial.turn.current); // the drag has come to rest
    return !d.moved;
  };
  const onBezelDown = (e: PointerEvent) => {
    if (e.button !== 0) return;
    const ctm = svg.current?.getScreenCTM();
    if (!ctm) return;
    e.preventDefault(); // no text selection or native drag, which would steal the pointer mid-turn
    endDrag();
    const centre = new DOMPoint(MAP.cx, MAP.cy).matrixTransform(ctm);
    const el = e.currentTarget as Element;
    try {
      el.setPointerCapture(e.pointerId);
    } catch {
      // the pointer is already gone; the window listeners below still end the drag
    }
    const onWindowUp = () => endDrag();
    window.addEventListener("pointerup", onWindowUp);
    window.addEventListener("pointercancel", onWindowUp);
    window.addEventListener("blur", onWindowUp);
    drag.current = {
      pointerId: e.pointerId, last: headingFrom(centre.x, centre.y, e.clientX, e.clientY), moved: false,
      cx: centre.x, cy: centre.y, el,
      unlisten: () => {
        window.removeEventListener("pointerup", onWindowUp);
        window.removeEventListener("pointercancel", onWindowUp);
        window.removeEventListener("blur", onWindowUp);
      },
    };
    setTip(null);
  };
  const onBezelMove = (e: PointerEvent) => {
    const d = drag.current;
    if (!d || d.pointerId !== e.pointerId) return;
    if (e.buttons === 0) {
      endDrag(); // the release was missed: never let the ring follow a mouse with no button held
      return;
    }
    const at = headingFrom(d.cx, d.cy, e.clientX, e.clientY);
    const delta = shortestTurn(d.last, at);
    if (!d.moved && Math.abs(delta) < 2) return;
    if (!d.moved) setDragging(true);
    d.moved = true;
    // measure from the last pointer heading so the turn stays continuous past +/-180
    dial.setTurn(dial.turn.current + delta);
    d.last = at;
  };
  const onBezelUp = (): boolean => endDrag();
  const onBezelKey = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 45 : 15;
    const keys: Record<string, number> = { ArrowRight: step, ArrowUp: step, ArrowLeft: -step, ArrowDown: -step };
    if (e.key in keys) dial.turnTo(dial.turn.current + keys[e.key], 300);
    else if (e.key === "Home") dial.turnTo(0);
    else return;
    e.preventDefault();
  };
  /** Turn the bezel so this role faces north. */
  const faceRole = (roleAngle: number) => dial.turnTo(-roleAngle);

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
      {/* starts hidden: it first fades in when the opening spin lands. The class prop never changes, so React leaves
          the imperative is-turning toggle alone. */}
      <p className="cp-facing is-turning" ref={facingEl}>{shownFacing}</p>
      <svg
        ref={svg}
        viewBox={`${-MAP_PAD.x} ${-MAP_PAD.y} ${MAP.size + 2 * MAP_PAD.x} ${MAP.size + 2 * MAP_PAD.y}`}
        className="cp-map"
        role="group"
        aria-label={`Compass map of the jobs you picked and suggested jobs, arranged by role: ${result.roles.map((r) => r.name).join(", ")}. Use the table view for the same information as a list.`}
      >
        {/* scope: fixed on north, its width set by what the bezel brings under it */}
        <path ref={scopePath} d={wedgePath(0, scopeHalfWidth(result.scope, turn), MAP.outer)} className="cp-scope" />
        {[ringBest, ringReach, ringStretch].map((r, i) => (
          <circle key={i} cx={MAP.cx} cy={MAP.cy} r={r} fill="none" style={{ stroke: "var(--cp-grid)", strokeWidth: 1 }} />
        ))}

        {/* the bezel: role spokes and degree ticks turn together */}
        <g ref={bezelG} transform={`rotate(${turn} ${MAP.cx} ${MAP.cy})`}>
          {result.roles.map((r) => {
            const [x0, y0] = polar(r.angle, 34);
            const [x1, y1] = polar(r.angle, MAP.outer);
            return <line key={r.name} x1={x0} y1={y0} x2={x1} y2={y1} style={{ stroke: "var(--cp-grid)", strokeWidth: 1, strokeDasharray: "2 4" }} />;
          })}
          {DEGREE_TICKS.map((d) => {
            const [x0, y0] = polar(d, MAP.outer);
            const [x1, y1] = polar(d, MAP.outer - (d % 30 === 0 ? 10 : 5));
            return <line key={d} x1={x0} y1={y0} x2={x1} y2={y1} style={{ stroke: "var(--cp-axis)", strokeWidth: 1 }} />;
          })}
          {result.roles.map((r) => {
            const [x0, y0] = polar(r.angle, MAP.outer);
            const [x1, y1] = polar(r.angle, MAP.outer + 10);
            return <line key={r.name} x1={x0} y1={y0} x2={x1} y2={y1} style={{ stroke: "var(--cp-axis)", strokeWidth: 2 }} />;
          })}
        </g>
        <circle cx={MAP.cx} cy={MAP.cy} r={MAP.outer} fill="none" style={{ stroke: "var(--cp-axis)", strokeWidth: 1 }} />
        <circle
          ref={slider}
          className={`cp-bezel${dragging ? " is-dragging" : ""}`}
          cx={MAP.cx}
          cy={MAP.cy}
          r={MAP.outer + 14}
          fill="none"
          strokeWidth={34}
          tabIndex={0}
          role="slider"
          aria-label="Turn the role ring"
          aria-valuemin={0}
          aria-valuemax={359}
          aria-valuenow={heading}
          aria-valuetext={`${view.facing}.`}
          onPointerDown={onBezelDown}
          onPointerMove={onBezelMove}
          onPointerUp={onBezelUp}
          onPointerCancel={onBezelUp}
          onLostPointerCapture={onBezelUp}
          onKeyDown={onBezelKey}
        />
        {/* each spoke's tick: hover for its name, click to turn it to north, drag to turn the ring */}
        <g ref={hitsG} transform={`rotate(${turn} ${MAP.cx} ${MAP.cy})`}>
          {result.roles.map((r) => {
            const [x0, y0] = polar(r.angle, MAP.outer - 6);
            const [x1, y1] = polar(r.angle, MAP.outer + 16);
            return (
              <line
                key={r.name}
                x1={x0} y1={y0} x2={x1} y2={y1}
                className="cp-spokehit"
                onPointerDown={onBezelDown}
                onPointerMove={onBezelMove}
                onPointerUp={() => { if (onBezelUp()) faceRole(r.angle); }}
                onPointerCancel={onBezelUp}
                onLostPointerCapture={onBezelUp}
              >
                <title>{`${r.name}. Click to face it, or drag to turn the ring.`}</title>
              </line>
            );
          })}
        </g>
        {/* labels follow the bezel but stay upright */}
        {result.roles.map((r, ri) => {
          const { x, y, anchor } = labelPlacement(r.angle + turn, lines[ri].length);
          return (
            <text
              key={r.name}
              ref={(el) => { labels.current[ri] = el; }}
              x={x}
              y={y}
              textAnchor={anchor}
              className={`cp-map__role${r.count ? "" : " is-empty"}${isMinor(result.roles, ri) ? " is-minor" : ""}${view.inScope[ri] === "1" ? " is-in-scope" : ""}`}
              onPointerDown={onBezelDown}
              onPointerMove={onBezelMove}
              onPointerUp={() => { if (onBezelUp()) faceRole(r.angle); }}
              onPointerCancel={onBezelUp}
              onLostPointerCapture={onBezelUp}
            >
              <title>{`${r.name}: ${r.count ? `${r.count} on the map` : "none of these jobs, shown for contrast"}. Click to face it, or drag to turn the ring.`}</title>
              {lines[ri].map((line, i) => (
                <tspan key={i} x={x} dy={i === 0 ? 0 : 13}>{line}</tspan>
              ))}
            </text>
          );
        })}

        {/* the fixed card: N/E/S/W never turn */}
        {CARDINALS.map(([label, d]) => {
          const [x, y] = polar(d, MAP.outer - 24);
          return <text key={label} x={x} y={y + 5} textAnchor="middle" className={`cp-map__cardinal${label === "N" ? " is-north" : ""}`}>{label}</text>;
        })}

        {/* the needle: always north, never moves */}
        <g
          style={{
            opacity: placed ? 1 : 0,
            transition: "opacity .6s .9s",
          }}
        >
          <line x1={MAP.cx} y1={MAP.cy + 18} x2={MAP.cx} y2={MAP.cy + 70} strokeWidth={3} strokeLinecap="round" style={{ stroke: "var(--cp-axis)" }} />
          <line x1={MAP.cx} y1={MAP.cy - 18} x2={MAP.cx} y2={MAP.cy - 150} strokeWidth={3} strokeLinecap="round" style={{ stroke: "var(--accent)" }} />
          <path d={`M${MAP.cx} ${MAP.cy - 168} l-9 20 h18 z`} style={{ fill: "var(--accent)" }} />
        </g>

        <circle cx={MAP.cx} cy={MAP.cy} r={22} style={{ fill: "var(--surface)", stroke: "var(--accent)", strokeWidth: 3 }} />
        <text x={MAP.cx} y={MAP.cy + 4} textAnchor="middle" className="cp-map__you">You</text>

        <g ref={dotsG} transform={`rotate(${turn} ${MAP.cx} ${MAP.cy})`}>
        {[...result.jobs.entries()]
          .sort(([, a], [, b]) => Number(activeGroup !== null && (activeGroup === "pick") === a.pick) - Number(activeGroup !== null && (activeGroup === "pick") === b.pick))
          .map(([i, j]) => { // the active kind is drawn last, so it sits on top; i stays the job's own index for the dial's mark refs
          const theme = result.themes.find((t) => t.id === j.themeId) ?? result.themes[0];
          const [x, y] = polar(j.angle, fitToRadius(j.fit));
          const visible = matches(j);
          const selected = selectedId === j.job.id;
          const color = themeColor(j.themeId);
          const front = activeGroup !== null && (activeGroup === "pick") === j.pick;
          const back = activeGroup !== null && !front;
          const dim = highlightId !== null && highlightId !== j.job.id;
          const lit = highlightId === j.job.id;
          // the ripple marks what is in focus: the active kind, or (with nothing active) the suggested roles
          return (
            <g
              key={j.job.id}
              className={`cp-dot${dim ? " is-dim" : ""}${lit ? " is-lit" : ""}${scopeJobs.has(j.job.id) ? "" : " is-out"}${back ? " is-back" : ""}${front ? " is-front" : ""}`}
              tabIndex={visible ? 0 : -1}
              role="button"
              aria-label={`${j.job.title} at ${j.job.company.name}, ${j.role}, ${theme.name}, fit ${j.fit} percent, ${TIER_NAME[tierOf(j.fit, stretch)]}${j.pick ? "" : ", suggested"}`}
              style={{
                transform: placed ? `translate(${x}px, ${y}px)` : `translate(${MAP.cx}px, ${MAP.cy}px)`,
                opacity: placed ? (visible ? 1 : 0.12) : 0,
                pointerEvents: visible && !back ? "auto" : "none",
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
              <g ref={(el) => { marks.current[i] = el; }} transform={`rotate(${-turn})`}>
              {(front || (activeGroup === null && !j.pick)) && <circle className="cp-halo" r={12} style={{ fill: color, opacity: 0.35 }} />}
              {j.pick ? (
                <rect className="cp-mark" x={-7} y={-7} width={14} height={14} rx={2} transform="rotate(45)" strokeWidth={selected ? 3 : 2} style={{ fill: color, stroke: selected ? "var(--text)" : "var(--surface)" }} />
              ) : (
                <circle className="cp-mark" r={8} strokeWidth={selected ? 3 : 2} style={{ fill: color, stroke: selected ? "var(--text)" : "var(--surface)" }} />
              )}
              </g>
            </g>
          );
        })}
        </g>
      </svg>

      {tip && (
        <div className="cp-tip" role="status" style={{ left: Math.min(tip.x, (wrap.current?.clientWidth ?? 400) - 250), top: tip.y }}>
          <b>{tip.job.job.title}</b>
          <i>{tip.job.job.company.name}{tip.job.pick ? "" : " · suggested"}</i>
          <div>{tip.job.role} · {result.themes.find((t) => t.id === tip.job.themeId)?.name} · {TIER_NAME[tierOf(tip.job.fit, stretch)]} · fit {tip.job.fit}%</div>
        </div>
      )}

      <div className="cp-legend">
        <span className="cp-legend__label">Color is the theme:</span>
        {result.themes.map((t) => (
          <span key={t.id}><i className="cp-key" style={{ background: themeColor(t.id) }} />{t.name}</span>
        ))}
        <span className="cp-legend__label">Shape:</span>
        <button type="button" className={`cp-legend__item${activeGroup === "pick" ? " is-on" : ""}`} aria-pressed={activeGroup === "pick"} title="Bring the jobs you picked to the front" onClick={() => toggleGroup("pick")}>
          <i className="cp-key cp-key--diamond" style={{ background: "var(--text-muted)" }} />A job you picked ({result.jobs.filter((j) => j.pick).length})
        </button>
        <button type="button" className={`cp-legend__item${activeGroup === "suggested" ? " is-on" : ""}`} aria-pressed={activeGroup === "suggested"} title="Bring the suggested roles to the front" onClick={() => toggleGroup("suggested")}>
          <i className="cp-key" style={{ background: "var(--text-muted)" }} />A suggested role ({result.jobs.filter((j) => !j.pick).length})
        </button>
        <span><i className="cp-key cp-key--scope" />Scope: what faces north (narrower means a tighter set of good fits)</span>
        <span><i className="cp-key cp-key--rings" />Rings, inside out: {TIER_NAME.best}, {TIER_NAME.reach}, {TIER_NAME.stretch}</span>
        <span className="cp-legend__hint">Click a shape to bring those jobs to the front (click again to clear). Drag the outer ring to turn it, or click a role to face it.</span>
      </div>
    </div>
  );
}
