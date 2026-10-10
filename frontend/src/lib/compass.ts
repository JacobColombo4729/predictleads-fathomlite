import type { CompassJob, CompassResult, CompassTheme, RequirementStatus } from "../types";

/** The Compass needs at least this many picks to find themes worth showing; the backend allows 2. */
export const COMPASS_MIN = 3;
/** The analysis stays fast and readable up to this many picks. */
export const COMPASS_MAX = 20;

export type Tier = "best" | "reach" | "stretch";
export const TIER_NAME: Record<Tier, string> = { best: "Best fit", reach: "Reach", stretch: "Stretch" };

/* ---------- compass geometry (SVG viewBox 0 0 640 640, you are at the centre) ---------- */

export const MAP = { size: 640, cx: 320, cy: 320, outer: 292 } as const;

/** Radius for a 0-100 fit: closer to the centre means a closer fit. */
export const fitToRadius = (fit: number): number => 60 + (100 - fit) * 2.3;

/** Compass heading in degrees (clockwise from north) and radius to SVG coordinates. */
export function polar(deg: number, r: number): [number, number] {
  const a = (deg * Math.PI) / 180;
  return [MAP.cx + r * Math.sin(a), MAP.cy - r * Math.cos(a)];
}

export function wedgePath(centerDeg: number, halfWidth: number, r: number): string {
  const [x0, y0] = polar(centerDeg - halfWidth, r);
  const [x1, y1] = polar(centerDeg + halfWidth, r);
  return `M${MAP.cx} ${MAP.cy} L${x0} ${y0} A${r} ${r} 0 0 1 ${x1} ${y1} Z`;
}

/* ---------- tiers: the stretch slider moves the Best fit / Reach / Stretch boundaries ---------- */

export function tierThresholds(stretch: number): { best: number; reach: number } {
  return { best: 82 - 0.16 * stretch, reach: 64 - 0.2 * stretch };
}

export function tierOf(fit: number, stretch: number): Tier {
  const t = tierThresholds(stretch);
  return fit >= t.best ? "best" : fit >= t.reach ? "reach" : "stretch";
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
