import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FeedbackSignal, Filters, Job, Profile, Recommendation, SearchScope, Stage, Tracker, TrackedJob } from "../types";
import { api, ApiError } from "../lib/api";
import { blankProfile, isBlankProfile } from "../lib/profile";

export interface Snapshot { id: string; date: string; jobCount: number }

type Boot = { status: "booting" } | { status: "ready" } | { status: "error"; message: string };

/**
 * Everything the app gets from the matching service: session, profile, ranked recommendations,
 * relevance feedback and saved jobs. The browser keeps no copy of record; the server session does.
 */
export function useLive(filters: Filters) {
  const [boot, setBoot] = useState<Boot>({ status: "booting" });
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [profile, setProfile] = useState<Profile>(blankProfile);
  const [profileId, setProfileId] = useState<string | null>(null);
  const [profileVersion, setProfileVersion] = useState(0);

  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [recsLoading, setRecsLoading] = useState(false);
  const [recsError, setRecsError] = useState<string | null>(null);
  const [ranker, setRanker] = useState<string>("");
  const [searchScope, setSearchScope] = useState<SearchScope>("all");
  const [feedbackApplied, setFeedbackApplied] = useState(false);
  const [ratings, setRatings] = useState<Record<string, FeedbackSignal>>({});

  const [tracker, setTracker] = useState<Tracker>({});
  const [savedItems, setSavedItems] = useState<Recommendation[]>([]);
  const noteTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  /* ---------- bootstrap: session, snapshot, profile ---------- */
  const bootstrap = useCallback(async () => {
    setBoot({ status: "booting" });
    try {
      const snap = await api.snapshot();
      setSnapshot(snap);
      // Reuse the existing server session (cookie); only create one when the server says there is none.
      let current;
      try {
        current = await api.currentProfile();
      } catch (e) {
        if (!(e instanceof ApiError && e.status === 401)) throw e;
        await api.ensureSession();
        current = await api.currentProfile();
      }
      if (!current) current = await api.createProfile(blankProfile);
      setProfile(current.profile);
      setProfileId(current.id);
      setProfileVersion(current.version);
      const [fb, saved] = await Promise.all([api.listFeedback(snap.id), api.listSaved(snap.id)]);
      setRatings(Object.fromEntries(fb.map((f) => [f.jobId, f.signal])));
      setTracker(Object.fromEntries(saved.map((s) => [s.recommendation.job.id, s.tracked])));
      setSavedItems(saved.map((s) => s.recommendation));
      setBoot({ status: "ready" });
    } catch (e) {
      setBoot({ status: "error", message: e instanceof Error ? e.message : "Could not start." });
    }
  }, []);

  useEffect(() => { void bootstrap(); }, [bootstrap]);

  /* ---------- recommendations (re-fetched on profile, filters or feedback changes) ---------- */
  const [refreshTick, setRefreshTick] = useState(0);
  const refresh = useCallback(() => setRefreshTick((t) => t + 1), []);
  const requestId = useRef(0);
  const filtersKey = JSON.stringify(filters);

  useEffect(() => {
    if (boot.status !== "ready" || !profileId || !snapshot) return;
    if (isBlankProfile(profile)) {
      // Nothing to match against yet: do not rank an empty profile.
      requestId.current++;
      setRecs([]);
      setRecsLoading(false);
      setRecsError(null);
      return;
    }
    const mine = ++requestId.current;
    setRecsLoading(true);
    setRecsError(null);
    const handle = setTimeout(async () => {
      try {
        const result = await api.recommendations(profileId, profileVersion, snapshot.id, JSON.parse(filtersKey) as Filters);
        if (mine !== requestId.current) return; // a newer request superseded this one
        setRecs(result.items);
        setRanker(result.ranker);
        setSearchScope(result.searchScope);
        setFeedbackApplied(result.feedback.applied);
      } catch (e) {
        if (mine !== requestId.current) return;
        setRecsError(e instanceof Error ? e.message : "Could not load recommendations.");
      } finally {
        if (mine === requestId.current) setRecsLoading(false);
      }
    }, 350);
    return () => clearTimeout(handle);
  }, [boot.status, profileId, profileVersion, snapshot, filtersKey, refreshTick, profile]);

  /* ---------- profile ---------- */
  const saveProfile = useCallback(async (next: Profile) => {
    if (!profileId) return;
    const updated = await api.updateProfile(profileId, profileVersion, next);
    setProfile(updated.profile);
    setProfileVersion(updated.version);
  }, [profileId, profileVersion]);

  /** Deletes the server session (profile, saved jobs, ratings) and starts again from a blank profile. */
  const startOver = useCallback(async () => {
    await api.resetSession();
    setRecs([]);
    setRatings({});
    setTracker({});
    setSavedItems([]);
    setProfile(blankProfile);
    setProfileId(null);
    setProfileVersion(0);
    await bootstrap();
  }, [bootstrap]);

  /* ---------- relevance feedback ---------- */
  const rate = useCallback(async (job: Job, signal: FeedbackSignal) => {
    if (!snapshot) return;
    setRatings((r) => ({ ...r, [job.id]: signal }));
    setRecs((list) => list.filter((x) => x.job.id !== job.id)); // rated jobs leave the list right away
    try {
      await api.setFeedback(job.id, snapshot.id, signal);
    } finally {
      refresh();
    }
  }, [snapshot, refresh]);

  const unrate = useCallback(async (jobId: string) => {
    if (!snapshot) return;
    setRatings(({ [jobId]: _gone, ...rest }) => rest);
    try {
      await api.clearFeedback(jobId, snapshot.id);
    } finally {
      refresh();
    }
  }, [snapshot, refresh]);

  const clearRatings = useCallback(async () => {
    if (!snapshot) return;
    const ids = Object.keys(ratings);
    setRatings({});
    await Promise.all(ids.map((id) => api.clearFeedback(id, snapshot.id).catch(() => undefined)));
    refresh();
  }, [snapshot, ratings, refresh]);

  /* ---------- saved jobs ---------- */
  const save = useCallback(async (job: Job) => {
    if (!snapshot || job.id in tracker) return;
    const entry: TrackedJob = { stage: "Saved", note: "", savedAt: new Date().toISOString() };
    setTracker((t) => ({ ...t, [job.id]: entry }));
    setSavedItems((items) => (items.some((i) => i.job.id === job.id) ? items : [{ job }, ...items]));
    try {
      await api.saveJob(job.id, snapshot.id);
    } catch (e) {
      setTracker(({ [job.id]: _gone, ...rest }) => rest);
      throw e;
    }
  }, [snapshot, tracker]);

  const remove = useCallback((jobId: string): TrackedJob | undefined => {
    if (!snapshot) return undefined;
    const prev = tracker[jobId];
    if (!prev) return undefined;
    setTracker(({ [jobId]: _gone, ...rest }) => rest);
    void api.removeSaved(jobId, snapshot.id).catch(() => undefined);
    return prev;
  }, [snapshot, tracker]);

  const restore = useCallback(async (jobId: string, entry: TrackedJob) => {
    if (!snapshot) return;
    setTracker((t) => (jobId in t ? t : { ...t, [jobId]: entry }));
    await api.saveJob(jobId, snapshot.id);
    await api.updateSaved(jobId, snapshot.id, { stage: entry.stage, note: entry.note });
  }, [snapshot]);

  const update = useCallback((jobId: string, patch: { stage?: Stage; note?: string }) => {
    if (!snapshot) return;
    setTracker((t) => (jobId in t ? { ...t, [jobId]: { ...t[jobId], ...patch } } : t));
    if (patch.note !== undefined) {
      clearTimeout(noteTimers.current[jobId]);
      noteTimers.current[jobId] = setTimeout(() => void api.updateSaved(jobId, snapshot.id, { note: patch.note }).catch(() => undefined), 600);
      if (patch.stage === undefined) return;
    }
    if (patch.stage !== undefined) void api.updateSaved(jobId, snapshot.id, { stage: patch.stage }).catch(() => undefined);
  }, [snapshot]);

  /** Lookup list for pages: current results plus saved jobs not in the current results. */
  const all = useMemo(() => {
    const ids = new Set(recs.map((r) => r.job.id));
    return [...recs, ...savedItems.filter((s) => !ids.has(s.job.id))];
  }, [recs, savedItems]);

  const counts = useMemo(() => {
    const values = Object.values(ratings);
    return { good: values.filter((v) => v === "good").length, bad: values.filter((v) => v === "bad").length };
  }, [ratings]);

  return {
    boot, retry: bootstrap, snapshot, profile, saveProfile, startOver, profileReady: profileId !== null, profileId, profileVersion,
    recs, recsLoading, recsError, ranker, searchScope, feedbackApplied, refresh,
    ratings, ratingCounts: counts, rate, unrate, clearRatings,
    tracker, all, save, remove, restore, update,
    isApiError: (e: unknown): e is ApiError => e instanceof ApiError,
  };
}
