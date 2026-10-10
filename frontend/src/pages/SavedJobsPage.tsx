import { useMemo, useState } from "react";
import type { Profile, Recommendation, Stage, TrackedJob } from "../types";
import { STAGES } from "../types";
import { formatSavedDate } from "../lib/jobs";
import { routeHref } from "../lib/router";
import { COMPASS_MAX, COMPASS_MIN } from "../lib/compass";
import JobDetails from "../components/JobDetails";
import StageSelect from "../components/StageSelect";

interface Row {
  item: Recommendation;
  tracked: TrackedJob;
}

interface Props {
  all: Recommendation[];
  tracker: Record<string, TrackedJob>;
  profile: Profile;
  snapshotId: string | null;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onRemove: (id: string) => void;
  onToggleSave: (item: Recommendation) => void;
  onStageChange: (id: string, s: Stage) => void;
  onNoteChange: (id: string, note: string) => void;
  /** Saved jobs ticked to send to the Compass, in the order they were ticked. */
  picked: string[];
  onTogglePick: (id: string) => void;
  /** Replace the whole selection (used by the select-all box). */
  onSetPicks: (ids: string[]) => void;
  onClearPicks: () => void;
  onSendToCompass: () => void;
}

export default function SavedJobsPage(p: Props) {
  const [query, setQuery] = useState("");
  const [stage, setStage] = useState<Stage | "all">("all");

  const rows: Row[] = useMemo(
    () =>
      p.all
        .filter((r) => r.job.id in p.tracker)
        .map((item) => ({ item, tracked: p.tracker[item.job.id] }))
        .sort((a, b) => b.tracked.savedAt.localeCompare(a.tracked.savedAt)),
    [p.all, p.tracker],
  );

  const counts = useMemo(() => {
    const c: Record<string, number> = { all: rows.length };
    STAGES.forEach((s) => (c[s] = rows.filter((r) => r.tracked.stage === s).length));
    return c;
  }, [rows]);

  const q = query.trim().toLowerCase();
  const visible = rows.filter(
    ({ item, tracked }) =>
      (stage === "all" || tracked.stage === stage) &&
      (!q || `${item.job.title} ${item.job.company.name} ${tracked.note}`.toLowerCase().includes(q)),
  );

  const selected = rows.find((r) => r.item.job.id === p.selectedId)?.item ?? null;

  const pickedSet = new Set(p.picked);
  const visibleIds = visible.map((r) => r.item.job.id);
  const allVisiblePicked = visibleIds.length > 0 && visibleIds.every((id) => pickedSet.has(id));
  const canSend = p.picked.length >= COMPASS_MIN;
  const toggleAllVisible = () =>
    p.onSetPicks(allVisiblePicked ? p.picked.filter((id) => !visibleIds.includes(id)) : [...p.picked, ...visibleIds.filter((id) => !pickedSet.has(id))]);

  return (
    <>
      <main className="browse" aria-label="Saved jobs">
        <header className="browse__head">
          <div>
            <h1>Saved Jobs</h1>
            <p className="browse__sub">Your own stages and notes. Tick the jobs that feel like you, then send them to the Compass.</p>
          </div>
          <div className="browse__actions">
            {rows.length > 0 && (
              <button
                type="button"
                className="btn btn--primary btn--lg"
                disabled={!canSend}
                onClick={p.onSendToCompass}
                title={canSend ? "Analyze the picked jobs in the Compass" : `Tick at least ${COMPASS_MIN} jobs first`}
              >
                Send to Compass{p.picked.length > 0 ? ` (${p.picked.length})` : ""}
              </button>
            )}
            <span className="demo-tag">Saved to your demo session</span>
          </div>
        </header>

        {rows.length === 0 ? (
          <div className="empty empty--page" role="status">
            <h3>No saved jobs yet</h3>
            <p>Save jobs from Discover to track them here.</p>
            <a className="btn btn--primary" href={routeHref("discover")}>Go to Discover Jobs</a>
          </div>
        ) : (
          <>
            <div className="filters">
              <label className="search">
                <span className="sr-only">Search saved jobs</span>
                <input
                  type="search"
                  value={query}
                  placeholder="Search saved jobs by title, company or note"
                  onChange={(e) => setQuery(e.target.value)}
                />
              </label>
              <div className="chips" role="group" aria-label="Filter by stage">
                {(["all", ...STAGES] as const).map((s) => (
                  <button
                    key={s}
                    type="button"
                    className="chip"
                    aria-pressed={stage === s}
                    onClick={() => setStage(s)}
                  >
                    {s === "all" ? "All" : s}<span className="chip__n">{counts[s]}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="resultbar" aria-live="polite">
              <strong>{visible.length} saved {visible.length === 1 ? "job" : "jobs"}</strong>
              {visible.length !== rows.length && <span className="muted"> of {rows.length}</span>}
            </div>

            {visible.length === 0 ? (
              <div className="empty" role="status">
                <h3>No saved jobs match</h3>
                <p>Try another search term or stage.</p>
                <button type="button" className="btn btn--primary" onClick={() => { setQuery(""); setStage("all"); }}>
                  Clear search and stage
                </button>
              </div>
            ) : (
              <div className="browse__list saved-scroll">
                <div className="tablewrap">
                  <table className="table">
                    <thead>
                      <tr>
                        <th scope="col" className="pickcol">
                          <input
                            type="checkbox"
                            className="pickbox"
                            checked={allVisiblePicked}
                            onChange={toggleAllVisible}
                            aria-label="Pick every job shown for the Compass"
                          />
                        </th>
                        <th scope="col">Job</th>
                        <th scope="col">Company</th>
                        <th scope="col">Location</th>
                        <th scope="col">Pay</th>
                        <th scope="col">My stage</th>
                        <th scope="col">Date saved</th>
                        <th scope="col"><span className="sr-only">Actions</span></th>
                      </tr>
                    </thead>
                    <tbody>
                      {visible.map(({ item, tracked }) => {
                        const { job } = item;
                        const sel = p.selectedId === job.id;
                        return (
                          <tr key={job.id} className={`${sel ? "row--selected" : ""}${pickedSet.has(job.id) ? " row--picked" : ""}`} onClick={() => p.onSelect(job.id)}>
                            <td className="pickcol">
                              <input
                                type="checkbox"
                                className="pickbox"
                                checked={pickedSet.has(job.id)}
                                onChange={() => p.onTogglePick(job.id)}
                                onClick={(e) => e.stopPropagation()}
                                aria-label={`Pick ${job.title} for the Compass`}
                              />
                            </td>
                            <td>
                              <button type="button" className="rowlink" aria-current={sel ? "true" : undefined} onClick={(e) => { e.stopPropagation(); p.onSelect(job.id); }}>
                                {job.title}
                              </button>
                              {tracked.note && <p className="row__note">Note: {tracked.note}</p>}
                            </td>
                            <td>{job.company.name}</td>
                            <td className={job.location ? "" : "muted"}>{job.location ?? "Not provided"}</td>
                            <td className={job.salary ? "nowrap" : "muted"}>{job.salary ?? "Not provided"}</td>
                            <td><StageSelect value={tracked.stage} label={`Stage for ${job.title}`} onChange={(s) => p.onStageChange(job.id, s)} /></td>
                            <td className="nowrap">{formatSavedDate(tracked.savedAt)}</td>
                            <td>
                              <button type="button" className="btn btn--ghost btn--sm" onClick={(e) => { e.stopPropagation(); p.onRemove(job.id); }} aria-label={`Remove ${job.title} from saved jobs`}>
                                Remove
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <ul className="savedcards">
                  {visible.map(({ item, tracked }) => {
                    const { job } = item;
                    return (
                      <li key={job.id} className={`card${p.selectedId === job.id ? " card--selected" : ""}${pickedSet.has(job.id) ? " card--picked" : ""}`}>
                        <label className="card__pick">
                          <input type="checkbox" className="pickbox" checked={pickedSet.has(job.id)} onChange={() => p.onTogglePick(job.id)} />
                          Pick for the Compass
                        </label>
                        <div>
                          <h3 className="card__title">{job.title}</h3>
                          <p className="card__company">{job.company.name}</p>
                        </div>
                        <p className="card__meta-line">
                          <span className={job.location ? "" : "muted"}>{job.location ?? "Location not provided"}</span>
                          <span className={job.salary ? "" : "muted"}>{job.salary ?? "Pay: Not provided"}</span>
                        </p>
                        {tracked.note && <p className="row__note">Note: {tracked.note}</p>}
                        <div className="savedcards__row">
                          <StageSelect value={tracked.stage} label={`Stage for ${job.title}`} onChange={(s) => p.onStageChange(job.id, s)} />
                          <span className="muted small">Saved {formatSavedDate(tracked.savedAt)}</span>
                        </div>
                        <div className="card__actions card__actions--full">
                          <button type="button" className="btn btn--ghost" onClick={() => p.onRemove(job.id)}>Remove</button>
                          <button type="button" className="btn btn--primary" onClick={() => p.onSelect(job.id)}>View details</button>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            <div className="sendbar" role="region" aria-label="Send picked jobs to the Compass">
              <div>
                <strong>{p.picked.length} picked</strong>
                <span className="sendbar__hint">
                  {p.picked.length === 0
                    ? " · tick the jobs that feel like you, then send them to the Compass"
                    : canSend
                      ? ` · up to ${COMPASS_MAX} work well`
                      : ` · pick at least ${COMPASS_MIN}`}
                </span>
              </div>
              <div className="cp-row" style={{ marginTop: 0 }}>
                <button type="button" className="btn btn--primary btn--lg" disabled={!canSend} onClick={p.onSendToCompass}>Send to Compass</button>
                {p.picked.length > 0 && <button type="button" className="btn btn--ghost" onClick={p.onClearPicks}>Clear</button>}
              </div>
            </div>
          </>
        )}
      </main>

      {selected && (
        <aside className="panel" aria-label="Selected saved job details">
          <JobDetails
            item={selected}
            tracked={p.tracker[selected.job.id]}
            profile={p.profile}
            snapshotId={p.snapshotId}
            onToggleSave={() => p.onToggleSave(selected)}
            onStageChange={(s) => p.onStageChange(selected.job.id, s)}
            onNoteChange={(n) => p.onNoteChange(selected.job.id, n)}
            onBack={() => p.onSelect(null)}
            backLabel="Back to saved jobs"
            backAlways
          />
        </aside>
      )}
    </>
  );
}
