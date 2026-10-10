import CompanyHiringBadge from "../signals/CompanyHiringBadge";
import { useLayoutEffect, useRef } from "react";
import type { CompassJob } from "../../types";
import { TIER_NAME, tierOf, type Tier } from "../../lib/compass";
import { prefersReducedMotion } from "./useAnimatedNumber";

interface Props {
  jobs: CompassJob[];
  themeFilter: string;
  stretch: number;
  selectedId: string | null;
  onSelect: (jobId: string) => void;
}

const TIERS: Tier[] = ["best", "reach", "stretch"];

/** Your picks and the suggested roles in three tiers by fit. Releasing the stretch slider brings in roles further from your picks; cards slide to their column. */
export default function TierBoard({ jobs, themeFilter, stretch, selectedId, onSelect }: Props) {
  const board = useRef<HTMLDivElement>(null);
  const rects = useRef(new Map<string, { left: number; top: number }>());

  const picks = jobs
    .filter((j) => themeFilter === "all" || j.themeId === themeFilter)
    .sort((a, b) => b.fit - a.fit);
  const byTier: Record<Tier, CompassJob[]> = { best: [], reach: [], stretch: [] };
  picks.forEach((j) => byTier[tierOf(j.fit, stretch)].push(j));

  // FLIP: remember where each card was, then animate from the old position to the new one after every render.
  useLayoutEffect(() => {
    const root = board.current;
    if (!root) return;
    const origin = root.getBoundingClientRect();
    const next = new Map<string, { left: number; top: number }>();
    root.querySelectorAll<HTMLElement>("[data-id]").forEach((card) => {
      const id = card.dataset.id!;
      const r = card.getBoundingClientRect();
      const pos = { left: r.left - origin.left, top: r.top - origin.top };
      next.set(id, pos);
      const before = rects.current.get(id);
      if (before && !prefersReducedMotion()) {
        const dx = before.left - pos.left;
        const dy = before.top - pos.top;
        if (Math.abs(dx) > 1 || Math.abs(dy) > 1) {
          card.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }], { duration: 480, easing: "cubic-bezier(.2,.8,.2,1)" });
        }
      }
    });
    rects.current = next;
  });

  return (
    <div className="cp-board" ref={board}>
      {TIERS.map((tier) => (
        <div key={tier} className="cp-col">
          <h3><span><i className={`cp-tiermark cp-tiermark--${tier}`} />{TIER_NAME[tier]}</span><small>{byTier[tier].length}</small></h3>
          {byTier[tier].length === 0 && <p className="cp-sub cp-col__empty">No jobs here at this setting.</p>}
          {byTier[tier].map((j) => (
            <div
              key={j.job.id}
              data-id={j.job.id}
              className={`cp-jcard${selectedId === j.job.id ? " is-sel" : ""}`}
              role="button"
              tabIndex={0}
              onClick={() => onSelect(j.job.id)}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(j.job.id); } }}
            >
              <div className="cp-jcard__t">{j.job.title}{!j.pick && <span className="cp-chip cp-chip--tag">Suggested</span>}</div>
              <div className="cp-jcard__c">{j.job.company.name}</div>
              <CompanyHiringBadge companyId={j.job.company.id} />
              <div className="cp-jcard__row">
                <div className="cp-meter"><i style={{ width: `${j.fit}%` }} /></div>
                <span>{j.fit}%</span>
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
