import type { CompassJob, CompassResult, CompassRole, CompassScope, CompassTheme, RequirementStatus } from "../types";

/** The Compass needs at least this many picks to find themes worth showing; the backend allows 2. */
export const COMPASS_MIN = 3;
/** The analysis stays fast and readable up to this many picks. */
export const COMPASS_MAX = 20;

export type Tier = "best" | "reach" | "stretch";
export const TIER_NAME: Record<Tier, string> = { best: "Best fit", reach: "Reach", stretch: "Stretch" };

/* ---------- compass geometry (SVG viewBox 0 0 640 640, you are at the centre) ---------- */

export const MAP = { size: 640, cx: 320, cy: 320, outer: 292 } as const;
/** Room around the 640 square for role labels outside the ring. */
export const MAP_PAD = { x: 130, y: 40 } as const;

/** Radius for a 0-100 fit: closer to the centre means a closer fit. */
export const fitToRadius = (fit: number): number => 60 + (100 - fit) * 2.3;

/** Compass heading in degrees (clockwise from north) and radius to SVG coordinates. */
export function polar(deg: number, r: number): [number, number] {
  const a = (deg * Math.PI) / 180;
  return [MAP.cx + r * Math.sin(a), MAP.cy - r * Math.cos(a)];
}

/** Pie slice from the centre, centred on a heading. */
export function wedgePath(centerDeg: number, halfWidth: number, r: number): string {
  const [x0, y0] = polar(centerDeg - halfWidth, r);
  const [x1, y1] = polar(centerDeg + halfWidth, r);
  return `M${MAP.cx} ${MAP.cy} L${x0} ${y0} A${r} ${r} 0 ${halfWidth > 90 ? 1 : 0} 1 ${x1} ${y1} Z`;
}

/** Signed shortest turn from `from` to `to`, in degrees (-180..180]. */
export const shortestTurn = (from: number, to: number): number => {
  const d = ((((to - from) % 360) + 540) % 360) - 180;
  return d === -180 ? 180 : d;
};

/* ---------- scope: the slice of the map that faces north (computed by the backend) ---------- */

/** Scope half-width with the bezel turned by `turn`, interpolated between the backend's precomputed headings. */
export function scopeHalfWidth(scope: CompassScope, turn: number): number {
  const n = scope.halfWidths.length;
  if (!n) return 60;
  const at = ((((-turn) % 360) + 360) % 360) / scope.step;
  const lo = Math.floor(at) % n;
  const t = at - Math.floor(at);
  return scope.halfWidths[lo] * (1 - t) + scope.halfWidths[(lo + 1) % n] * t;
}

/**
 * How far north sits between two neighbouring spokes, as a share of half the gap: 0 is on a spoke, 1 is midway.
 * Below onSpoke it reads as that role, below strong as mostly that role, below moderate as a lean, else between.
 */
export const LEAN = { onSpoke: 0.15, strong: 0.45, moderate: 0.8 } as const;

/**
 * Plain words for what faces north. Every phrasing starts "Facing", so turning the ring only changes the end:
 * "Facing Data Scientist" / "Facing mostly Data Scientist, with a little Statistician" /
 * "Facing Data Scientist, leaning toward Statistician" / "Facing between Data Scientist and Statistician".
 */
export function facing(roles: CompassRole[], turn: number): string {
  if (!roles.length) return "";
  if (roles.length === 1) return `Facing ${roles[0].name}`;
  const offs = roles.map((r) => ({ name: r.name, off: shortestTurn(0, r.angle + turn) }));
  const nearest = offs.reduce((a, b) => (Math.abs(b.off) < Math.abs(a.off) ? b : a));
  const lean = Math.abs(nearest.off) / (180 / roles.length);
  if (lean < LEAN.onSpoke) return `Facing ${nearest.name}`;
  // the spoke on the other side of north
  const side = offs.filter((o) => Math.sign(o.off) === -Math.sign(nearest.off));
  if (!side.length) return `Facing ${nearest.name}`;
  const other = side.reduce((a, b) => (Math.abs(b.off) < Math.abs(a.off) ? b : a));
  if (lean < LEAN.strong) return `Facing mostly ${nearest.name}, with a little ${other.name}`;
  if (lean < LEAN.moderate) return `Facing ${nearest.name}, leaning toward ${other.name}`;
  const [left, right] = nearest.off < 0 ? [nearest, other] : [other, nearest];
  return `Facing between ${left.name} and ${right.name}`;
}

/** Word-wrap a role name for a spoke label; a "/" stays at the end of a line rather than starting the next. */
export function wrapLabel(text: string, width = 18): string[] {
  const lines: string[] = [];
  for (const word of text.split(" ")) {
    const last = lines[lines.length - 1];
    if (last !== undefined && (word === "/" || `${last} ${word}`.length <= width)) lines[lines.length - 1] = `${last} ${word}`;
    else lines.push(word);
  }
  return lines;
}

/* ---------- tiers: fixed cutoffs on the 0-100 fit; the stretch slider changes which roles are suggested, not the cutoffs ---------- */

export const TIER_CUTOFFS = { best: 75, reach: 55 } as const;

export function tierThresholds(_stretch?: number): { best: number; reach: number } {
  return { best: TIER_CUTOFFS.best, reach: TIER_CUTOFFS.reach };
}

export function tierOf(fit: number, _stretch?: number): Tier {
  return fit >= TIER_CUTOFFS.best ? "best" : fit >= TIER_CUTOFFS.reach ? "reach" : "stretch";
}

/* ---------- themes and statuses ---------- */

/** Categorical slots are assigned in a fixed order, so a theme keeps its colour when others are filtered out. */
export const themeColor = (themeId: string): string => {
  const index = Math.min(2, Math.max(0, Number(themeId.replace(/\D/g, "")) - 1));
  return `var(--cp-s${index + 1})`;
};

export const STATUS_LABEL: Record<RequirementStatus, string> = {
  supported: "Supported",
  partial: "Partial",
  unknown: "Not mentioned",
};

export function themeOf(result: CompassResult, job: CompassJob): CompassTheme {
  return result.themes.find((t) => t.id === job.themeId) ?? result.themes[0];
}

export const requirementShortName = (text: string, limit = 90): string => {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length <= limit ? clean : `${clean.slice(0, limit - 1).trimEnd()}…`;
};
