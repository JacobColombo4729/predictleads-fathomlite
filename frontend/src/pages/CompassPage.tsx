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
import GapSection from "../components/compass/GapSection";
import type { GapBatch } from "../state/useGapBatch";
import { themeColor } from "../lib/compass";

interface Props {
  state: CompassState;
  currentProfileVersion: number;
  savedIds: Set<string>;
  onSave: (job: Job) => void;
  onRerun: () => void;
  /** Called when the stretch slider is released, so the suggested roles can reach further out. */
  onStretch: (stretch: number) => void;
  gap: GapBatch;
  snapshotReady: boolean;
}

function defaultSelection(result: CompassResult): string | null {
  const picks = result.jobs.filter((j) => j.pick).sort((a, b) => b.fit - a.fit);
  return picks[0]?.job.id ?? result.jobs[0]?.job.id ?? null;
}

export default function CompassPage({ state, currentProfileVersion, savedIds, onSave, onRerun, onStretch, gap, snapshotReady }: Props) {
  // If an analysis already exists when the page opens, skip the reveal; a fresh run plays it.
  const [revealDone, setRevealDone] = useState(state.status === "ready");
  const [shown, setShown] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [highlightId, setHighlightId] = useState<string | null>(null);
  const [themeFilter, setThemeFilter] = useState("all");
  const [added, setAdded] = useState<CompassJob[]>([]);
  const [notice, setNotice] = useState<{ text: string; kind: "tiers" | "saved" } | null>(null);
  const [stretch, setStretch] = useState(0);
  const [table, setTable] = useState(false);
  const [scopeIds, setScopeIds] = useState<string[]>([]);

  useEffect(() => {
    if (state.status === "loading") {
      setRevealDone(false);
      setSelectedId(null);
      setThemeFilter("all");
      setTable(false);
      setStretch(0);
      setAdded([]);
      setNotice(null);
    }
  }, [state.status]);

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 7000);
    return () => clearTimeout(t);
  }, [notice]);

  const addRole = useCallback((job: CompassJob) => {
    setAdded((list) => (list.some((x) => x.job.id === job.job.id) ? list : [...list, job]));
    setNotice({ text: `Added “${job.job.title}” (${job.job.company.name}) to Your fit, in tiers.`, kind: "tiers" });
  }, []);
  const removeRole = useCallback((jobId: string) => {
    setAdded((list) => list.filter((x) => x.job.id !== jobId));
    setNotice(null);
  }, []);
  const saveRole = useCallback((job: CompassJob) => {
    onSave(job.job);
    setNotice({ text: `Saved “${job.job.title}” (${job.job.company.name}) to your Saved jobs.`, kind: "saved" });
  }, [onSave]);
  const showTiers = () => { document.getElementById("cp-tiers")?.scrollIntoView({ behavior: "smooth", block: "start" }); setNotice(null); };

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
  // Roles the person added stay on the board and map even after the stretch slider replaces the suggestions.
  const addedIds = new Set(added.map((j) => j.job.id));
  const viewResult: CompassResult | null = result ? { ...result, jobs: [...result.jobs.filter((j) => !addedIds.has(j.job.id)), ...added] } : null;
  const listResult: CompassResult | null = result ? { ...result, jobs: result.jobs.filter((j) => j.pick || !addedIds.has(j.job.id)) } : null;
  const selected: CompassJob | null = viewResult ? viewResult.jobs.find((j) => j.job.id === (selectedId ?? defaultSelection(viewResult))) ?? null : null;
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
                  <JobTable result={viewResult!} themeFilter={themeFilter} stretch={stretch} onSelect={setSelectedId} onHighlight={setHighlightId} />
                ) : (
                  <CompassMap
                    result={viewResult!}
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
                result={viewResult!}
                job={selected}
                stretch={stretch}
                saved={selected ? savedIds.has(selected.job.id) : false}
                onSave={(j) => onSave(j.job)}
                added={selected ? addedIds.has(selected.job.id) : false}
                onAdd={addRole}
                onRemove={removeRole}
                scopeJobs={table ? null : scopeIds.map((id) => viewResult!.jobs.find((j) => j.job.id === id)).filter((j) => j !== undefined)}
                onSelect={setSelectedId}
                onClear={() => setSelectedId(null)}
                onHighlight={setHighlightId}
              />
            </div>
          </section>

          <section className="cp-block" id="cp-tiers">
            <div className="cp-card">
              <header className="cp-cardhead">
                <div>
                  <h2>Your fit, in tiers</h2>
                  <p className="cp-sub">
                    Slide right to see roles further from what you picked, then let go and the suggested roles update. Your own picks stay where they are.
                    Best fit is 75 and up, Reach is 55 to 74, Stretch is below 55: how similar a role is to your profile, not a chance of being hired.
                  </p>
                </div>
              </header>
              <div className="cp-range">
                <span>Stay close to my experience</span>
                <input type="range" min={0} max={100} value={stretch} aria-label="How much stretch are you open to" onChange={(e) => setStretch(Number(e.target.value))}
                  onPointerUp={(e) => onStretch(Number((e.target as HTMLInputElement).value))}
                  onKeyUp={(e) => { if (e.key.startsWith("Arrow") || e.key === "Home" || e.key === "End") onStretch(Number((e.target as HTMLInputElement).value)); }} />
                <span>Open to a stretch</span>
              </div>
              {state.status === "ready" && state.updating && <p className="cp-sub" role="status">Finding roles at this stretch…</p>}
              {state.status === "ready" && state.notice && <p className="field__error" role="alert">{state.notice}</p>}
              <TierBoard jobs={viewResult!.jobs} themeFilter={themeFilter} stretch={stretch} selectedId={selected?.job.id ?? null} onSelect={setSelectedId} addedIds={addedIds} onAdd={addRole} onRemove={removeRole} onHighlight={setHighlightId} savedIds={savedIds} onSave={saveRole} />
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
            <GapSection picks={[...result.jobs.filter((j) => j.pick), ...added]} gap={gap} snapshotReady={snapshotReady} />
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
                  <p className="cp-sub">Different titles, similar work. Hover a card to find it on the compass. Press “Add to my tiers” to move a role up into Your fit, in tiers; it stays there when you move the stretch slider, and it is included in the skills gap analysis.</p>
                </div>
              </header>
              <Suggestions result={listResult!} selectedId={selected?.job.id ?? null} onSelect={setSelectedId} onHighlight={setHighlightId} onAdd={addRole} savedIds={savedIds} onSave={saveRole} />
            </div>
          </section>

          <footer className="cp-foot">
            {result.limits.map((l) => <p key={l}>{l}</p>)}
          </footer>
        </div>
      )}
      {notice && (
        <div className="cp-toast" role="status">
          <span>{notice.text}</span>
          {notice.kind === "tiers"
            ? <button type="button" className="cp-toast__btn" onClick={showTiers}>Show me</button>
            : <a className="cp-toast__btn" href={routeHref("saved")}>View Saved</a>}
          <button type="button" className="cp-toast__x" aria-label="Dismiss" onClick={() => setNotice(null)}>×</button>
        </div>
      )}
    </main>
  );
}
