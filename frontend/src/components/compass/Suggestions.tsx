import type { CompassJob, CompassResult } from "../../types";
import { requirementShortName, themeColor } from "../../lib/compass";

interface Props {
  result: CompassResult;
  selectedId: string | null;
  onSelect: (jobId: string) => void;
  onHighlight: (jobId: string | null) => void;
}

function blurb(job: CompassJob): string {
  const first = job.why[0];
  return first
    ? `Your “${first.evidence}” lines up with “${requirementShortName(first.requirement, 100)}”.`
    : "It sits close to the jobs you picked and to your profile, under a title you didn’t pick.";
}

/** Listings you did not pick that sit near your picks and your profile. Hovering a card lights its diamond on the map. */
export default function Suggestions({ result, selectedId, onSelect, onHighlight }: Props) {
  const suggestions = result.jobs.filter((j) => !j.pick);
  if (suggestions.length === 0) return <p className="cp-sub">No further suggestions for this set of picks.</p>;
  return (
    <div className="cp-sugs">
      {suggestions.map((j) => {
        const theme = result.themes.find((t) => t.id === j.themeId);
        const builds = j.builds.map((id) => result.requirements.find((r) => r.id === id)).filter((r) => r !== undefined);
        return (
          <div
            key={j.job.id}
            className={`cp-sug${selectedId === j.job.id ? " is-sel" : ""}`}
            role="button"
            tabIndex={0}
            onClick={() => onSelect(j.job.id)}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(j.job.id); } }}
            onMouseEnter={() => onHighlight(j.job.id)}
            onMouseLeave={() => onHighlight(null)}
            onFocus={() => onHighlight(j.job.id)}
            onBlur={() => onHighlight(null)}
          >
            <span className="cp-sug__badge">Hidden fit</span>
            <div className="cp-sug__t">{j.job.title}</div>
            <div className="cp-sug__c">{j.job.company.name}{theme ? ` · ${theme.name}` : ""}</div>
            <p>{blurb(j)}</p>
            <div className="cp-chips">
              {theme && <span className="cp-chip"><i className="cp-key" style={{ background: themeColor(theme.id) }} />Fit {j.fit}%</span>}
              {builds.map((r) => <span key={r.id} className="cp-chip" title={r.text}>Builds: {requirementShortName(r.text, 36)}</span>)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
