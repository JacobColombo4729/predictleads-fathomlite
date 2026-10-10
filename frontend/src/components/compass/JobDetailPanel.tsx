import CompanyHiringBadge from "../signals/CompanyHiringBadge";
import { useEffect, useState } from "react";
import type { CompassJob, CompassResult } from "../../types";
import { STATUS_LABEL, TIER_NAME, requirementShortName, themeColor, tierOf } from "../../lib/compass";
import { ExternalIcon } from "../Icons";

interface Props {
  result: CompassResult;
  job: CompassJob | null;
  stretch: number;
  saved: boolean;
  onSave: (job: CompassJob) => void;
  added: boolean;
  onAdd: (job: CompassJob) => void;
  onRemove: (jobId: string) => void;
  /** Jobs inside the map's scope wedge, best fit first; null when the map is not showing. */
  scopeJobs: CompassJob[] | null;
  onSelect: (jobId: string) => void;
  onClear: () => void;
  onHighlight: (jobId: string | null) => void;
}

/**
 * Why a pick or suggestion fits, and where you would stretch. Every quoted line is verbatim from your profile or the
 * listing. With nothing selected it lists the jobs inside the map's scope, which change as you turn the ring.
 */
export default function JobDetailPanel({ result, job, stretch, saved, onSave, added, onAdd, onRemove, scopeJobs, onSelect, onClear, onHighlight }: Props) {
  const [meter, setMeter] = useState(0);
  useEffect(() => {
    setMeter(0);
    if (!job) return;
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setMeter(job.fit)));
    return () => cancelAnimationFrame(id);
  }, [job]);

  if (!job && scopeJobs) {
    return (
      <div className="cp-card cp-detail">
        <div className="cp-eyebrow">In scope · {scopeJobs.length}</div>
        <p className="cp-sub">The jobs inside the highlighted slice. Turn the ring to explore another direction.</p>
        {scopeJobs.length === 0 ? (
          <p className="cp-empty">Nothing faces north here. Turn the ring toward the dots.</p>
        ) : (
          <ul className="cp-scopelist">
            {scopeJobs.map((j) => (
              <li key={j.job.id}>
                <button type="button" onClick={() => onSelect(j.job.id)} onMouseEnter={() => onHighlight(j.job.id)} onMouseLeave={() => onHighlight(null)} onFocus={() => onHighlight(j.job.id)} onBlur={() => onHighlight(null)}>
                  <span className="cp-scopelist__t">{j.job.title}{j.pick ? "" : <i> · suggested</i>}</span>
                  <span className="cp-scopelist__c">{[j.job.company.name, j.role].filter(Boolean).join(" · ")}</span>
                  <span className="cp-scopelist__fit">{TIER_NAME[tierOf(j.fit, stretch)]} · {j.fit}%</span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }
  if (!job) return <div className="cp-card cp-detail"><p className="cp-empty">Select a dot to see why it fits and where you’d stretch.</p></div>;

  const theme = result.themes.find((t) => t.id === job.themeId);
  const reqs = (ids: string[]) => ids.map((id) => result.requirements.find((r) => r.id === id)).filter((r) => r !== undefined);
  const buildsOrStretch = job.pick ? reqs(job.stretch) : reqs(job.builds);

  return (
    <div className="cp-card cp-detail" aria-live="polite">
      {scopeJobs && <button type="button" className="cp-back" onClick={onClear}>← Jobs in scope</button>}
      <div className="cp-eyebrow">{job.pick ? "Picked job" : added ? "Added role" : "Suggested job"}</div>
      <h3>{job.job.title}</h3>
      <p className="cp-sub">{job.job.company.name}{job.job.location ? ` · ${job.job.location}` : ""}</p>
      <CompanyHiringBadge companyId={job.job.company.id} />
      <div className="cp-chips">
        {theme && <span className="cp-chip"><i className="cp-key" style={{ background: themeColor(theme.id) }} />{theme.name}</span>}
        <span className="cp-chip">{TIER_NAME[tierOf(job.fit, stretch)]}</span>
      </div>

      <div className="cp-sub">Fit to your profile</div>
      <div className="cp-meter" role="img" aria-label={`Fit ${job.fit} out of 100`}><i style={{ width: `${meter}%` }} /></div>
      <p className="cp-sub cp-gap-sm">{job.fit}% similarity to your profile, scaled against all listings. This is a similarity score, not a chance of being hired.</p>

      <div className="cp-eyebrow">Why it fits</div>
      {job.why.length === 0 ? (
        <p className="cp-sub cp-gap-sm">No single line of your profile stands out against this listing’s requirements.</p>
      ) : (
        <ul className="cp-why">
          {job.why.map((w) => (
            <li key={`${w.evidence}-${w.requirement}`}>
              Your “{w.evidence}” <span className="cp-muted">({w.source})</span> lines up with “{requirementShortName(w.requirement, 110)}”
            </li>
          ))}
        </ul>
      )}

      {buildsOrStretch.length > 0 && (
        <>
          <div className="cp-eyebrow">{job.pick ? "Where you’d stretch" : "Builds toward"}</div>
          <div className="cp-chips cp-gap-sm">
            {buildsOrStretch.map((r) => (
              <span key={r.id} className="cp-chip" title={r.text}>
                <i className={`cp-key cp-key--${r.status}`} />
                {requirementShortName(r.text, 44)} · {STATUS_LABEL[r.status].toLowerCase()}
              </span>
            ))}
          </div>
        </>
      )}

      <div className="cp-row">
        {job.job.sourceUrl && (
          <a className="btn" href={job.job.sourceUrl} target="_blank" rel="noopener noreferrer">
            <ExternalIcon />View listing
          </a>
        )}
        {!job.pick && (added
          ? <button type="button" className="btn" onClick={() => onRemove(job.job.id)}>Remove from my tiers</button>
          : <button type="button" className="btn btn--primary" onClick={() => onAdd(job)}>+ Add to my tiers</button>)}
        <button type="button" className="btn" disabled={saved} onClick={() => onSave(job)}>
          {saved ? "Saved" : "Save job"}
        </button>
      </div>
    </div>
  );
}
