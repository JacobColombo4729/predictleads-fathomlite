import type { FeedbackSignal, Filters, ListedRange, Profile, Recommendation, SearchScope, Stage, TrackedJob } from "../types";
import FilterBar from "../components/FilterBar";
import JobCard from "../components/JobCard";
import JobDetails from "../components/JobDetails";
import MlNote from "../components/MlNote";
import { isBlankProfile } from "../lib/profile";

// Career changers start at entry level, so that is the default; the filter bar changes it.
export const defaultFilters: Filters = { query: "", location: "", description: "any", levels: ["entry"] };
const allFilters: Filters = { ...defaultFilters, levels: [] };

interface Props {
  recs: Recommendation[];
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  profile: Profile;
  snapshotId: string | null;
  snapshotDate: string | null;
  listedRange?: ListedRange | null;
  ranker: string;
  searchScope: SearchScope;
  feedbackApplied: boolean;
  ratingCounts: { good: number; bad: number };
  onClearRatings: () => void;
  onRate: (item: Recommendation, signal: FeedbackSignal) => void;
  filters: Filters;
  onFiltersChange: (f: Filters) => void;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onViewDetails: (id: string) => void;
  onBackToList: () => void;
  tracker: Record<string, TrackedJob>;
  onToggleSave: (item: Recommendation) => void;
  onStageChange: (id: string, s: Stage) => void;
  onNoteChange: (id: string, note: string) => void;
}

const rankerName: Record<string, string> = {
  tfidf_bge_rrf_equal: "keyword + meaning-based ranking (TF-IDF and BGE fused)",
  tfidf_cosine: "keyword ranking (TF-IDF)",
};

export default function DiscoverPage(p: Props) {
  const selected = p.recs.find((r) => r.job.id === p.selectedId) ?? p.recs[0] ?? null;
  const rated = p.ratingCounts.good + p.ratingCounts.bad;

  return (
    <>
      <main className="browse" aria-label="Discover jobs">
        <header className="browse__head">
          <div>
            <h1>Discover Jobs</h1>
            <p className="browse__sub">For {p.profile.name}{p.profile.headline ? ` · ${p.profile.headline}` : ""}</p>
          </div>
          <span className="demo-tag" title="Listings come from one historical snapshot">
            Historical listings{p.snapshotDate ? ` · ${p.snapshotDate}` : ""} · not verified as open
          </span>
        </header>

        <FilterBar filters={p.filters} onChange={p.onFiltersChange} listedRange={p.listedRange} />

        <div className="resultbar" aria-live="polite">
          <strong>{p.loading && p.recs.length === 0 ? "Finding jobs…" : `${p.recs.length} ${p.recs.length === 1 ? "job" : "jobs"}`}</strong>
          <span className="muted">
            {" "}· Ranked with {rankerName[p.ranker] ?? "the matching service"}. The number is the order, not a match percentage.
            {p.loading && p.recs.length > 0 ? " Updating…" : ""}
          </span>
        </div>

        <MlNote id="ranker" variant={p.ranker === "combined" || p.ranker === "ml_rank_queue" ? p.ranker : undefined} />

        {p.filters.query.trim() && p.searchScope !== "all" && (
          <p className="muted small searchnote" role="status">
            {p.searchScope === "title"
              ? <>Showing jobs with “{p.filters.query.trim()}” in the title, ranked by fit to your profile.</>
              : <>Few job titles match “{p.filters.query.trim()}”, so this also searches descriptions and tags.</>}
          </p>
        )}

        <section className="feedbackbar" aria-label="Your ratings">
          <p>
            <strong>Tell it what fits.</strong> Mark jobs “Good match” or “Not for me” and the list adapts: it moves toward the jobs you liked
            and away from the ones you rejected. Rated jobs are hidden from new results. This adapts your session only; it does not retrain the model.
          </p>
          <p className="feedbackbar__state">
            {rated === 0 ? "No ratings yet." : (
              <>
                <span>👍 {p.ratingCounts.good}</span> <span>👎 {p.ratingCounts.bad}</span>
                {p.feedbackApplied ? " · results adapted" : ""}{" "}
                <button type="button" className="linkbtn" onClick={p.onClearRatings}>Clear my ratings</button>
              </>
            )}
          </p>
        </section>
        <MlNote id="feedback" />

        {p.error && (
          <p className="notice" role="alert">
            {p.error}{" "}
            <button type="button" className="linkbtn" onClick={p.onRetry}>Try again</button>
          </p>
        )}

        <div className="browse__list">
          {isBlankProfile(p.profile) ? (
            <div className="empty" role="status">
              <h3>Create your profile to see matches</h3>
              <p>Discover ranks listings against your profile. Add your experience, skills and the roles you want, or use Magic fill with a resume, then save.</p>
              <a className="btn btn--primary" href="#/profile">Go to My Profile</a>
            </div>
          ) : p.recs.length === 0 && !p.loading && !p.error ? (
            <div className="empty" role="status">
              <h3>No jobs match these filters</h3>
              <p>Add more job levels, clear the search words or location, or rate fewer jobs, and try again.</p>
              <button type="button" className="btn btn--primary" onClick={() => p.onFiltersChange(allFilters)}>Clear filters</button>
            </div>
          ) : (
            p.recs.map((item) => (
              <JobCard
                key={item.job.id}
                item={item}
                selected={selected?.job.id === item.job.id}
                saved={item.job.id in p.tracker}
                onSelect={() => p.onSelect(item.job.id)}
                onViewDetails={() => p.onViewDetails(item.job.id)}
                onToggleSave={() => p.onToggleSave(item)}
                onRate={(signal) => p.onRate(item, signal)}
              />
            ))
          )}
        </div>
      </main>

      <aside className="panel" aria-label="Selected job details">
        <JobDetails
          item={selected}
          tracked={selected ? p.tracker[selected.job.id] : undefined}
          profile={p.profile}
          snapshotId={p.snapshotId}
          onToggleSave={() => selected && p.onToggleSave(selected)}
          onStageChange={(s) => selected && p.onStageChange(selected.job.id, s)}
          onNoteChange={(n) => selected && p.onNoteChange(selected.job.id, n)}
          onBack={p.onBackToList}
        />
      </aside>
    </>
  );
}
