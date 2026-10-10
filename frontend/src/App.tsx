import { useEffect, useState } from "react";
import type { FeedbackSignal, Filters, Recommendation, Stage } from "./types";
import { isBlankProfile } from "./lib/profile";
import { useRoute } from "./lib/router";
import { useLive } from "./state/useLive";
import SideNav from "./components/SideNav";
import Toast, { type ToastState } from "./components/Toast";
import DiscoverPage, { defaultFilters } from "./pages/DiscoverPage";
import SavedJobsPage from "./pages/SavedJobsPage";
import ProfilePage from "./pages/ProfilePage";

export default function App() {
  const route = useRoute();

  // Discover state lives here so it survives navigation.
  const [filters, setFilters] = useState<Filters>(defaultFilters);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mobileView, setMobileView] = useState<"list" | "detail">("list");
  const [savedSelectedId, setSavedSelectedId] = useState<string | null>(null);
  const [toast, setToast] = useState<ToastState | null>(null);

  const live = useLive(filters);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 8000);
    return () => clearTimeout(t);
  }, [toast]);

  const savedCount = Object.keys(live.tracker).length;

  const toggleSave = (item: Recommendation) => {
    if (item.job.id in live.tracker) {
      const prev = live.remove(item.job.id);
      if (prev) setToast({ message: `Removed "${item.job.title}" from saved jobs.`, onUndo: () => void live.restore(item.job.id, prev) });
    } else {
      live.save(item.job).catch(() => setToast({ message: "Could not save that job. Please try again." }));
    }
  };

  const rate = (item: Recommendation, signal: FeedbackSignal) => {
    live.rate(item.job, signal).catch(() => setToast({ message: "Could not record that rating. Please try again." }));
    setToast({
      message: signal === "good"
        ? `Marked "${item.job.title}" as a good match. Showing more like it.`
        : `Marked "${item.job.title}" as not for you. Showing fewer like it.`,
      onUndo: () => void live.unrate(item.job.id),
    });
  };

  const setStage = (id: string, s: Stage) => live.update(id, { stage: s });
  const setNote = (id: string, note: string) => live.update(id, { note });

  if (live.boot.status !== "ready") {
    return (
      <div className="boot" role="status">
        {live.boot.status === "booting" ? (
          <p>Connecting to the matching service…</p>
        ) : (
          <>
            <h1>Can't reach the matching service</h1>
            <p>{live.boot.message}</p>
            <p className="muted">Start the backend (run_demo_backend.ps1 in the project folder). It takes a few minutes to load the job data the first time.</p>
            <button type="button" className="btn btn--primary" onClick={() => void live.retry()}>Try again</button>
          </>
        )}
      </div>
    );
  }

  const panelOpen = route === "saved" && savedSelectedId !== null && savedSelectedId in live.tracker;
  const shellClass =
    route === "discover" ? `shell--discover shell--${mobileView}`
    : route === "saved" ? `shell--saved ${panelOpen ? "shell--detail shell--with-panel" : "shell--list"}`
    : "shell--profile";

  return (
    <div className={`shell ${shellClass}`}>
      <SideNav route={route} savedCount={savedCount} />

      {route === "discover" && (
        <DiscoverPage
          recs={live.recs}
          loading={live.recsLoading}
          error={live.recsError}
          onRetry={live.refresh}
          profile={live.profile}
          snapshotId={live.snapshot?.id ?? null}
          snapshotDate={live.snapshot?.date ?? null}
          ranker={live.ranker}
          feedbackApplied={live.feedbackApplied}
          ratingCounts={live.ratingCounts}
          onClearRatings={() => void live.clearRatings()}
          onRate={rate}
          filters={filters}
          onFiltersChange={setFilters}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onViewDetails={(id) => { setSelectedId(id); setMobileView("detail"); }}
          onBackToList={() => setMobileView("list")}
          tracker={live.tracker}
          onToggleSave={toggleSave}
          onStageChange={setStage}
          onNoteChange={setNote}
        />
      )}
      {route === "saved" && (
        <SavedJobsPage
          all={live.all}
          tracker={live.tracker}
          profile={live.profile}
          snapshotId={live.snapshot?.id ?? null}
          selectedId={savedSelectedId}
          onSelect={setSavedSelectedId}
          onRemove={(id) => {
            const item = live.all.find((r) => r.job.id === id);
            if (item) toggleSave(item);
          }}
          onToggleSave={toggleSave}
          onStageChange={setStage}
          onNoteChange={setNote}
        />
      )}
      {route === "profile" && (
        <ProfilePage
          profile={live.profile}
          edited={!isBlankProfile(live.profile)}
          recovered={false}
          onSave={live.saveProfile}
          onReset={live.startOver}
        />
      )}

      {toast && <Toast toast={toast} onDismiss={() => setToast(null)} />}
    </div>
  );
}
