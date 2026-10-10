import { useCallback, useEffect, useState } from "react";
import type { CompassJob, CompassResult, Job } from "../types";
import type { CompassState } from "../state/useCompass";
import { routeHref } from "../lib/router";
import Reveal from "../components/compass/Reveal";
import CompassMap from "../components/compass/CompassMap";
import JobDetailPanel from "../components/compass/JobDetailPanel";
import JobTable from "../components/compass/JobTable";
import TierBoard from "../components/compass/TierBoard";
import RequirementBars from "../components/compass/RequirementBars";
import NextSteps from "../components/compass/NextSteps";
import Suggestions from "../components/compass/Suggestions";
import { themeColor } from "../lib/compass";

interface Props {
  state: CompassState;
  currentProfileVersion: number;
  savedIds: Set<string>;
  onSave: (job: Job) => void;
  onRerun: () => void;
}

function defaultSelection(result: CompassResult): string | null {
  const picks = result.jobs.filter((j) => j.pick).sort((a, b) => b.fit - a.fit);
  return picks[0]?.job.id ?? result.jobs[0]?.job.id ?? null;
}

export default function CompassPage({ state, currentProfileVersion, savedIds, onSave, onRerun }: Props) {
  // If an analysis already exists when the page opens, skip the reveal; a fresh run plays it.
  const [revealDone, setRevealDone] = useState(state.status === "ready");
  const [shown, setShown] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [highlightId, setHighlightId] = useState<string | null>(null);
  const [themeFilter, setThemeFilter] = useState("all");
  const [stretch, setStretch] = useState(50);
  const [table, setTable] = useState(false);
  const [scopeIds, setScopeIds] = useState<string[]>([]);

  useEffect(() => {
    if (state.status === "loading") {
      setRevealDone(false);
      setSelectedId(null);
      setThemeFilter("all");
      setTable(false);
    }
  }, [state.status]);

  const finished = useCallback(() => {
    setRevealDone(true);
    setShown((n) => n + 1);
  }, []);

  if (state.status === "idle") {
    return (
      <main className="cp" aria-label="Compass">
        <header className="cp-head"><div><div className="cp-eyebrow">Compass</div><h1>Your career compass</h1></div></header>
        <div className="cp-card cp-start">
          <h2>Nothing to analyze yet</h2>
          <p className="cp-sub">
            Pick the jobs that feel like “you” on the Saved page and send them here. The Compass finds what they have in common,
            how your profile lines up, and other roles you might not have thought of.
          </p>
          <a className="btn btn--primary" href={routeHref("saved")}>Go to Saved jobs</a>
        </div>
      </main>
    );
  }

  const showReveal = state.status !== "ready" || !revealDone;
  const result = state.status === "ready" ? state.result : null;
  const stale = state.status === "ready" && state.profileVersion !== currentProfileVersion;
  const selected: CompassJob | null = result ? result.jobs.find((j) => j.job.id === (selectedId ?? defaultSelection(result))) ?? null : null;
  const picks = result?.picksAnalyzed ?? state.jobIds.length;

  return (
    <main className="cp" aria-label="Compass">
      <header className="cp-head">
        <div>
          <div className="cp-eyebrow">Compass</div>
          <h1>Your career compass</h1>
          <p className="cp-sub">Based on the {picks} {picks === 1 ? "job" : "jobs"} you sent from Saved.</p>
        </div>
        <div className="cp-row">
          <a className="btn" href={routeHref("saved")}>Back to Saved</a>
          <button type="button" className="btn" onClick={onRerun} disabled={state.status === "loading"}>Replay analysis</button>
        </div>
      </header>

      {stale && (
        <p className="notice notice--compact" role="status">
          Your profile changed after this analysis. <button type="button" className="linkbtn" onClick={onRerun}>Re-run it</button> to use the latest version.
        </p>
      )}

      {showReveal && (
        <Reveal
          key={state.status === "error" ? "error" : "run"}
          jobCount={state.jobIds.length}
          ready={state.status === "ready"}
          error={state.status === "error" ? state.message : null}
          onFinished={finished}
          onRetry={onRerun}
          onBack={() => { window.location.hash = routeHref("saved"); }}
        />
      )}

      {result && !showReveal && (
        <div key={shown}>
          <div className="cp-tiles cp-fade">
            <div className="cp-tile"><b>{result.stats.jobs}</b><span>jobs analyzed{result.duplicatesCollapsed > 0 ? ` (${result.duplicatesCollapsed} repost${result.duplicatesCollapsed === 1 ? "" : "s"} merged)` : ""}</span></div>
            <div className="cp-tile"><b>{result.stats.themes}</b><span>{result.stats.themes === 1 ? "theme" : "themes"} found</span></div>
            <div className="cp-tile"><b>{result.stats.groupsWithEvidence} of {result.stats.requirementGroups}</b><span>requirement groups with evidence</span></div>
            <div className="cp-tile"><b>{result.stats.suggestions}</b><span>roles you didn’t pick</span></div>
          </div>

          <section className="cp-block">
            <div className="cp-grid">
              <div className="cp-card">
                <header className="cp-cardhead">
                  <div>
                    <h2>Where your picks sit</h2>
                    <p className="cp-sub">Closer to the center means a closer match to your profile. Direction is the kind of role; turn the ring to point the scope at a direction.</p>
                  </div>
                  <button type="button" className="chip" aria-pressed={table} onClick={() => setTable((t) => !t)}>{table ? "View as map" : "View as table"}</button>
                </header>
                {result.themes.length > 1 && (
                  <div className="cp-filters" role="group" aria-label="Filter by theme">
                    <button type="button" aria-pressed={themeFilter === "all"} onClick={() => setThemeFilter("all")}>All themes</button>
                    {result.themes.map((t) => (
                      <button key={t.id} type="button" aria-pressed={themeFilter === t.id} onClick={() => setThemeFilter(t.id)}>
                        <i className="cp-key" style={{ background: themeColor(t.id) }} />{t.name}
                      </button>
                    ))}
                  </div>
                )}
                {table ? (
                  <JobTable result={result} themeFilter={themeFilter} stretch={stretch} onSelect={setSelectedId} />
                ) : (
                  <CompassMap
                    result={result}
                    themeFilter={themeFilter}
                    selectedId={selected?.job.id ?? null}
                    highlightId={highlightId}
                    stretch={stretch}
                    onSelect={setSelectedId}
                    onScopeChange={setScopeIds}
                  />
                )}
              </div>
              <JobDetailPanel
                result={result}
                job={selected}
                stretch={stretch}
                saved={selected ? savedIds.has(selected.job.id) : false}
                onSave={(j) => onSave(j.job)}
                scopeJobs={table ? null : scopeIds.map((id) => result.jobs.find((j) => j.job.id === id)).filter((j) => j !== undefined)}
                onSelect={setSelectedId}
                onClear={() => setSelectedId(null)}
              />
            </div>
          </section>

          <section className="cp-block">
            <div className="cp-card">
              <header className="cp-cardhead">
                <div>
                  <h2>Your fit, in tiers</h2>
                  <p className="cp-sub">Slide to decide how much stretch you’re open to. Cards move between tiers and the rings move on the map.</p>
                </div>
              </header>
              <div className="cp-range">
                <span>Stay close to my experience</span>
                <input type="range" min={0} max={100} value={stretch} aria-label="How much stretch are you open to" onChange={(e) => setStretch(Number(e.target.value))} />
                <span>Open to a stretch</span>
              </div>
              <TierBoard jobs={result.jobs} themeFilter={themeFilter} stretch={stretch} selectedId={selected?.job.id ?? null} onSelect={setSelectedId} />
            </div>
          </section>

          <section className="cp-block">
            <div className="cp-card">
              <header className="cp-cardhead">
                <div>
                  <h2>What your picks ask for</h2>
                  <p className="cp-sub">Requirement lines from your {picks} jobs, grouped by meaning, with the closest line of your own profile.</p>
                </div>
              </header>
              <RequirementBars requirements={result.requirements} picks={picks} />
            </div>
          </section>

          <section className="cp-block">
            <div className="cp-card">
              <header className="cp-cardhead">
                <div>
                  <h2>Suggested next steps</h2>
                  <p className="cp-sub">Ordered by how often a requirement appears in your picks and how little your profile shows it. Tap a status to track it.</p>
                </div>
              </header>
              <NextSteps steps={result.nextSteps} picks={picks} />
            </div>
          </section>

          <section className="cp-block">
            <div className="cp-card">
              <header className="cp-cardhead">
                <div>
                  <h2>Roles you didn’t pick but might fit</h2>
                  <p className="cp-sub">Different titles, similar work. Hover a card to find it on the compass.</p>
                </div>
              </header>
              <Suggestions result={result} selectedId={selected?.job.id ?? null} onSelect={setSelectedId} onHighlight={setHighlightId} />
            </div>
          </section>

          <footer className="cp-foot">
            {result.limits.map((l) => <p key={l}>{l}</p>)}
          </footer>
        </div>
      )}
    </main>
  );
}
