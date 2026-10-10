import { useEffect, useState } from "react";
import type { CompassRequirement, RequirementStatus } from "../../types";
import { STATUS_LABEL } from "../../lib/compass";

interface Props {
  requirements: CompassRequirement[];
  picks: number;
}

type Filter = "all" | RequirementStatus;
const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "supported", label: "Supported" },
  { key: "partial", label: "Partial" },
  { key: "unknown", label: "Not mentioned" },
];

export function StatusIcon({ status }: { status: RequirementStatus }) {
  if (status === "supported") {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r="7" fill="var(--cp-good)" />
        <path d="M4.6 8.2l2.3 2.3 4.5-4.8" fill="none" stroke="#fff" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  if (status === "partial") {
    return (
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r="6.5" fill="none" stroke="var(--cp-warn)" strokeWidth="1.6" />
        <path d="M8 1.5a6.5 6.5 0 010 13z" fill="var(--cp-warn)" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="6.5" fill="none" stroke="var(--text-muted)" strokeWidth="1.6" strokeDasharray="2.4 2" />
      <path d="M6.2 6.3a1.9 1.9 0 113 1.5c-.7.5-1.2.8-1.2 1.6M8 11.6v.1" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

/** What your picks ask for (how many of them mention it) next to the closest line of your own profile. */
export default function RequirementBars({ requirements, picks }: Props) {
  const [filter, setFilter] = useState<Filter>("all");
  const [grown, setGrown] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setGrown(true)));
    return () => cancelAnimationFrame(id);
  }, []);

  const visible = requirements.filter((r) => filter === "all" || r.status === filter);
  if (requirements.length === 0) {
    return <p className="cp-sub">These jobs didn’t share enough requirement text to group. Try picking a few more.</p>;
  }

  return (
    <>
      <div className="cp-filters" role="group" aria-label="Filter requirements">
        {FILTERS.map((f) => (
          <button key={f.key} type="button" aria-pressed={filter === f.key} onClick={() => setFilter(f.key)}>{f.label}</button>
        ))}
      </div>
      <div>
        {visible.map((r) => (
          <div className="cp-req" key={r.id}>
            <div>
              <div className="cp-req__name">{r.text}</div>
              <div className="cp-req__ev">
                {r.evidence ? <>“{r.evidence.text}” — {r.evidence.source}</> : "Nothing in your profile mentions this yet."}
              </div>
            </div>
            <div>
              <div className="cp-lab"><span>In your picks</span><b>{r.n} of {picks}</b></div>
              <div className="cp-bar cp-bar--freq"><i style={{ width: grown ? `${(r.n / picks) * 100}%` : 0 }} /></div>
            </div>
            <div>
              <div className="cp-lab"><span>Closest evidence</span><b>{r.coverage}%</b></div>
              <div className="cp-bar cp-bar--cov"><i style={{ width: grown ? `${r.coverage}%` : 0 }} /></div>
              <div className={`cp-status cp-status--${r.status}`}>
                <StatusIcon status={r.status} />
                <span>{STATUS_LABEL[r.status]}</span>
                {r.status === "unknown" && <span className="cp-muted">· unknown, not a verdict</span>}
                {r.verify && <span className="cp-muted">· check this yourself</span>}
              </div>
            </div>
          </div>
        ))}
        {visible.length === 0 && <p className="cp-sub">Nothing in this group.</p>}
      </div>
    </>
  );
}
