import type { FeedbackSignal, Recommendation } from "../types";
import { formatDate, historicalStatusText } from "../lib/jobs";
import { BookmarkIcon, BuildingIcon, CalendarIcon, CashIcon, PinIcon, ThumbDownIcon, ThumbUpIcon } from "./Icons";

interface Props {
  item: Recommendation;
  selected: boolean;
  saved: boolean;
  onSelect: () => void;
  onViewDetails: () => void;
  onToggleSave: () => void;
  onRate: (signal: FeedbackSignal) => void;
}

/** Only call a result "similar" when the embedding similarity to the liked job is reasonably high. */
const SIMILAR_MIN = 0.78;

export default function JobCard({ item, selected, saved, onSelect, onViewDetails, onToggleSave, onRate }: Props) {
  const { job } = item;
  const similar = item.similarTo && item.similarTo.similarity >= SIMILAR_MIN ? item.similarTo : null;
  return (
    <article
      className={`card${selected ? " card--selected" : ""}`}
      aria-current={selected ? "true" : undefined}
      onClick={onSelect}
    >
      <header className="card__head">
        <div className="card__titles">
          <h3 className="card__title">{job.title}</h3>
          <p className="card__company">{job.company.name}</p>
        </div>
        {item.rank !== undefined && <span className="rank" title="Retrieval order, not a match percentage">#{item.rank}</span>}
      </header>

      <dl className="card__meta">
        <div><PinIcon /><dt className="sr-only">Location</dt><dd className={job.location ? "" : "muted"}>{job.location ?? "Location not provided"}</dd></div>
        <div><BuildingIcon /><dt className="sr-only">Work arrangement</dt><dd className="muted">Arrangement not provided</dd></div>
        <div><CashIcon /><dt className="sr-only">Salary</dt><dd className={job.salary ? "" : "muted"}>{job.salary ?? "Salary: Not provided"}</dd></div>
      </dl>

      {similar && (
        <p className="card__similar">More like “{similar.title ?? "a job you liked"}”, which you marked as a good match.</p>
      )}
      {job.description && <p className="card__why">{job.description.replace(/\s+/g, " ").slice(0, 200)}{job.description.length > 200 ? "…" : ""}</p>}

      <footer className="card__foot">
        <div className="card__status">
          <span className="obs"><CalendarIcon />Observed {formatDate(job.observedOn)}</span>
          <span className={`status status--${job.historicalStatus}`}>
            Status in snapshot: {historicalStatusText[job.historicalStatus]}. Not verified as open.
          </span>
        </div>
        <div className="card__actions">
          <button
            type="button"
            className="btn btn--ghost btn--good"
            title="Show me more jobs like this"
            onClick={(e) => { e.stopPropagation(); onRate("good"); }}
          >
            <ThumbUpIcon />Good match
          </button>
          <button
            type="button"
            className="btn btn--ghost btn--bad"
            title="Show me fewer jobs like this"
            onClick={(e) => { e.stopPropagation(); onRate("bad"); }}
          >
            <ThumbDownIcon />Not for me
          </button>
          <button
            type="button"
            className={`btn btn--ghost${saved ? " btn--saved" : ""}`}
            aria-pressed={saved}
            onClick={(e) => { e.stopPropagation(); onToggleSave(); }}
          >
            <BookmarkIcon filled={saved} />{saved ? "Saved" : "Save"}
          </button>
          <button
            type="button"
            className="btn btn--primary"
            onClick={(e) => { e.stopPropagation(); onViewDetails(); }}
          >
            View details
          </button>
        </div>
      </footer>
    </article>
  );
}
