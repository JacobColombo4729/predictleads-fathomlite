import { participantId } from "./research";
import type {
  GapBatchResult,
  GapEvidence,
  GapJobResult,
  SignalCompany,
  SignalPattern,
  SignalsOverview,
  SignalsQuery,
  CompassResult,
  SearchScope,
  EntryKind,
  FeedbackReason,
  FeedbackSignal,
  Filters,
  HistoricalStatus,
  Job,
  Profile,
  Recommendation,
  Stage,
  TrackedJob,
} from "../types";

/** Requests go to /api, which the Vite dev server proxies to the backend so the session cookie stays same-origin. */
const BASE = "/api/v1";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown, headers: Record<string, string> = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      credentials: "include",
      headers: { ...(body === undefined ? {} : { "Content-Type": "application/json" }), ...(participantId() ? { "X-Participant-Id": participantId()! } : {}), ...headers },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "network_error", "Could not reach the matching service. Is the backend running?");
  }
  if (response.status === 204) return undefined as T;
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    // non-JSON body
  }
  if (!response.ok) {
    const err = (payload as { error?: { code?: string; message?: string } } | null)?.error;
    throw new ApiError(response.status, err?.code ?? "http_error", err?.message ?? `Request failed (${response.status}).`);
  }
  return payload as T;
}

/* ---------- raw API shapes (only the fields the UI reads) ---------- */

interface ApiJob {
  job_id: string;
  title: string;
  company_summary: { company_id?: string | null; name: string | null; domain: string | null } | null;
  location: string | null;
  salary_raw: unknown;
  description?: string | null;
  description_excerpt: string | null;
  source_url: string | null;
  observed_on: string;
  source_status: { raw: string | null; state: string };
}

interface ApiProfile {
  profile_id: string;
  version: number;
  profile: {
    name: string;
    headline: string;
    experience: { id: string; kind?: EntryKind; title: string; organization: string | null; description: string }[];
    skills: { id: string; name: string; evidence: string }[];
    desired_roles: string[];
    location: string | null;
    arrangements: ("Remote" | "Hybrid" | "On-site")[];
    constraints: string | null;
  };
}

/* ---------- mappers ---------- */

function statusFor(state: string): HistoricalStatus {
  return state === "closed" ? "not_in_latest_snapshot" : "unknown_status";
}

export function jobFromApi(j: ApiJob): Job {
  const salary = typeof j.salary_raw === "string" && j.salary_raw.trim() ? j.salary_raw : null;
  return {
    id: j.job_id,
    title: j.title,
    company: {
      id: j.company_summary?.company_id ?? null,
      name: j.company_summary?.name?.trim() || "Company name not provided",
      domain: j.company_summary?.domain ?? null,
      description: null,
      industry: null,
      size: null,
      headquarters: null,
    },
    location: typeof j.location === "string" && j.location.trim() ? j.location : null,
    workArrangement: "Not provided",
    salary,
    description: j.description ?? j.description_excerpt,
    sourceUrl: j.source_url ?? "",
    observedOn: j.observed_on,
    historicalStatus: statusFor(j.source_status?.state ?? "unknown"),
  };
}

export function profileFromApi(p: ApiProfile): { profile: Profile; id: string; version: number } {
  const x = p.profile;
  return {
    id: p.profile_id,
    version: p.version,
    profile: {
      name: x.name,
      headline: x.headline,
      experience: x.experience.map((e) => ({ id: e.id, kind: e.kind ?? "job", title: e.title, organization: e.organization ?? "", description: e.description })),
      skills: x.skills.map((s) => ({ id: s.id, name: s.name, evidence: s.evidence })),
      desiredRoles: x.desired_roles,
      location: x.location ?? "",
      arrangements: x.arrangements,
      constraints: x.constraints ?? "",
    },
  };
}

export function profileToApi(p: Profile) {
  return {
    name: p.name,
    headline: p.headline,
    experience: p.experience.map((e) => ({ id: e.id, kind: e.kind, title: e.title, organization: e.organization || null, description: e.description })),
    skills: p.skills.map((s) => ({ id: s.id, name: s.name, evidence: s.evidence })),
    desired_roles: p.desiredRoles,
    location: p.location || null,
    arrangements: p.arrangements,
    constraints: p.constraints || null,
  };
}

/* ---------- endpoints ---------- */

const signalsQuery = (q: SignalsQuery): string => {
  const p = new URLSearchParams();
  if (q.pattern) p.set("pattern", q.pattern);
  if (q.family) p.set("family", q.family);
  if (q.q) p.set("q", q.q);
  p.set("sort", q.sort);
  p.set("limit", String(q.limit));
  p.set("offset", String(q.offset));
  return p.toString();
};

export const api = {
  signalsOverview: () => request<SignalsOverview>("GET", "/signals/overview"),
  signalsPatterns: async () => (await request<{ patterns: Record<string, SignalPattern> }>("GET", "/signals/patterns")).patterns,
  signalsCompanies: (q: SignalsQuery) => request<{ total: number; companies: SignalCompany[] }>("GET", `/signals/companies?${signalsQuery(q)}`),
  signalsCompany: (id: string) => request<SignalCompany>("GET", `/signals/companies/${encodeURIComponent(id)}`),

  async ensureSession(): Promise<void> {
    await request("POST", "/demo-sessions");
  },

  /** Deletes this browser's demo session on the server: profile, saved jobs, ratings and cached assessments. */
  async resetSession(): Promise<void> {
    await request("DELETE", "/demo-sessions/current");
  },

  async currentProfile(): Promise<{ profile: Profile; id: string; version: number } | null> {
    try {
      return profileFromApi(await request<ApiProfile>("GET", "/profiles/current"));
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }
  },

  async createProfile(p: Profile) {
    return profileFromApi(await request<ApiProfile>("POST", "/profiles", profileToApi(p)));
  },

  async updateProfile(id: string, version: number, p: Profile) {
    return profileFromApi(await request<ApiProfile>("PATCH", `/profiles/${id}`, profileToApi(p), { "If-Match": `"${version}"` }));
  },

  /** Resume "magic fill": sends the resume text to the backend's resume reader. Returns suggestions only; nothing is saved. */
  async extractFromResume(source: { text: string } | { pdfBase64: string }): Promise<
    | { state: "unavailable" }
    | {
        state: "completed";
        dropped: number;
        profile: {
          name: string; headline: string; location: string;
          experience: { kind: EntryKind; title: string; organization: string; description: string }[];
          skills: { name: string; evidence: string }[];
        };
      }
  > {
    return request("POST", "/profile-extraction", "text" in source ? { resume_text: source.text } : { resume_pdf_base64: source.pdfBase64 });
  },

  async snapshot(): Promise<{ id: string; date: string; jobCount: number }> {
    const r = await request<{ snapshot_id: string; date: string; job_count: number }>("GET", "/snapshots/current");
    return { id: r.snapshot_id, date: r.date, jobCount: r.job_count };
  },

  async recommendations(
    profileId: string,
    profileVersion: number,
    snapshotId: string,
    filters: Filters,
    limit = 25,
  ): Promise<{ items: Recommendation[]; feedback: { good: number; bad: number; applied: boolean }; ranker: string; searchScope: SearchScope }> {
    const apiFilters: Record<string, string> = {};
    if (filters.query.trim()) apiFilters.query = filters.query.trim();
    if (filters.location.trim()) apiFilters.location = filters.location.trim();
    if (filters.description !== "any") apiFilters.description = filters.description;
    const r = await request<{
      ranking: { ranker_id: string };
      feedback: { good: number; bad: number; applied: boolean };
      search?: { query: string; scope: SearchScope };
      items: { rank: number; job: ApiJob; exploration?: { source: "keyword" | "embedding" | "random" }; similar_to_liked?: { job_id: string; title: string | null; similarity: number } }[];
    }>("POST", `/profiles/${profileId}/recommendations`, {
      snapshot_id: snapshotId,
      profile_version: profileVersion,
      limit,
      filters: apiFilters,
      explore: true, // a few "something different" listings so ratings can compare the keyword and embedding methods
    });
    return {
      ranker: r.ranking.ranker_id,
      feedback: r.feedback,
      searchScope: r.search?.scope ?? "all",
      items: r.items.map((it) => ({
        job: jobFromApi(it.job),
        rank: it.rank,
        exploration: it.exploration ?? null,
        similarTo: it.similar_to_liked
          ? { jobId: it.similar_to_liked.job_id, title: it.similar_to_liked.title, similarity: it.similar_to_liked.similarity }
          : null,
      })),
    };
  },

  /** AI skills gap analysis for the Compass picks: ONE request to OpenAI covering every job, when the server has it enabled. */
  async compassGapAnalysis(snapshotId: string, jobIds: string[]): Promise<GapBatchResult> {
    const current = await api.currentProfile();
    if (!current) throw new ApiError(404, "profile_not_found", "Save your profile first.");
    const r = await request<{
      assessment_state: "completed" | "unavailable"; candidate_evidence: GapEvidence[]; jobs: Record<string, GapJobResult>; failed: number;
    }>("POST", `/profiles/${current.id}/compass/gap-analysis`, { snapshot_id: snapshotId, profile_version: current.version, job_ids: jobIds });
    return r.assessment_state === "completed"
      ? { state: "completed", candidateEvidence: r.candidate_evidence, jobs: r.jobs, failed: r.failed }
      : { state: "unavailable" };
  },

  /** Compass analysis of 2-20 picked jobs against the saved profile. `stretch` (0-100) sets how far the suggested roles reach. Local embeddings only; nothing leaves the machine. */
  async compass(profileId: string, profileVersion: number, snapshotId: string, jobIds: string[], stretch = 0): Promise<CompassResult> {
    const r = await request<{
      picks_analyzed: number;
      duplicates_collapsed: number;
      themes: { id: string; name: string; count: number; angle: number; half_width: number; mean_fit: number }[];
      strongest_theme_id: string;
      jobs: {
        job: ApiJob; pick: boolean; theme_id: string; fit: number; offset: number;
        why: { evidence: string; source: string; requirement: string }[]; stretch?: string[]; builds?: string[];
      }[];
      requirements: {
        id: string; text: string; n: number; coverage: number; status: "supported" | "partial" | "unknown";
        evidence: { text: string; source: string; matched_requirement: string } | null; verify?: boolean; job_ids: string[];
      }[];
      next_steps: { requirement_id: string; text: string; n: number; coverage: number; status: "supported" | "partial" | "unknown"; verify?: boolean; action: string }[];
      stats: { jobs: number; themes: number; requirement_groups: number; groups_with_evidence: number; suggestions: number; sentences_read: number };
      method: { limits: string[] };
    }>("POST", `/profiles/${profileId}/compass`, { snapshot_id: snapshotId, profile_version: profileVersion, job_ids: jobIds, stretch });
    return {
      picksAnalyzed: r.picks_analyzed,
      duplicatesCollapsed: r.duplicates_collapsed,
      themes: r.themes.map((t) => ({ id: t.id, name: t.name, count: t.count, angle: t.angle, halfWidth: t.half_width, meanFit: t.mean_fit })),
      strongestThemeId: r.strongest_theme_id,
      jobs: r.jobs.map((j) => ({
        job: jobFromApi(j.job), pick: j.pick, themeId: j.theme_id, fit: j.fit, offset: j.offset, why: j.why,
        stretch: j.stretch ?? [], builds: j.builds ?? [],
      })),
      requirements: r.requirements.map((q) => ({
        id: q.id, text: q.text, n: q.n, coverage: q.coverage, status: q.status, verify: q.verify ?? false, jobIds: q.job_ids,
        evidence: q.evidence ? { text: q.evidence.text, source: q.evidence.source, matchedRequirement: q.evidence.matched_requirement } : null,
      })),
      nextSteps: r.next_steps.map((s) => ({ requirementId: s.requirement_id, text: s.text, n: s.n, coverage: s.coverage, status: s.status, verify: s.verify ?? false, action: s.action })),
      stats: {
        jobs: r.stats.jobs, themes: r.stats.themes, requirementGroups: r.stats.requirement_groups,
        groupsWithEvidence: r.stats.groups_with_evidence, suggestions: r.stats.suggestions, sentencesRead: r.stats.sentences_read,
      },
      limits: r.method.limits,
    };
  },

  async jobDetails(jobId: string, snapshotId: string): Promise<Job> {
    return jobFromApi(await request<ApiJob>("GET", `/jobs/${encodeURIComponent(jobId)}?snapshot_id=${encodeURIComponent(snapshotId)}`));
  },

  async setFeedback(jobId: string, snapshotId: string, signal: FeedbackSignal, reason?: FeedbackReason, note?: string): Promise<void> {
    await request("PUT", "/feedback", { job_id: jobId, snapshot_id: snapshotId, signal, ...(reason ? { reason } : {}), ...(note ? { note } : {}) });
  },

  async clearFeedback(jobId: string, snapshotId: string): Promise<void> {
    await request("DELETE", `/feedback/${encodeURIComponent(snapshotId)}/${encodeURIComponent(jobId)}`);
  },

  async listFeedback(snapshotId: string): Promise<{ jobId: string; signal: FeedbackSignal }[]> {
    const r = await request<{ items: { job_id: string; signal: FeedbackSignal }[] }>("GET", `/feedback?snapshot_id=${encodeURIComponent(snapshotId)}`);
    return r.items.map((i) => ({ jobId: i.job_id, signal: i.signal }));
  },

  async listSaved(snapshotId: string): Promise<{ recommendation: Recommendation; tracked: TrackedJob }[]> {
    const r = await request<{ items: { job_id: string; stage: Stage; note: string; saved_at: string; job: ApiJob | null }[] }>(
      "GET",
      `/saved-jobs?snapshot_id=${encodeURIComponent(snapshotId)}`,
    );
    return r.items
      .filter((i) => i.job !== null)
      .map((i) => ({
        recommendation: { job: jobFromApi(i.job as ApiJob) },
        tracked: { stage: i.stage, note: i.note, savedAt: i.saved_at },
      }));
  },

  async saveJob(jobId: string, snapshotId: string): Promise<void> {
    await request("POST", "/saved-jobs", { job_id: jobId, snapshot_id: snapshotId });
  },

  async updateSaved(jobId: string, snapshotId: string, patch: { stage?: Stage; note?: string }): Promise<void> {
    await request("PATCH", `/saved-jobs/${encodeURIComponent(snapshotId)}/${encodeURIComponent(jobId)}`, patch);
  },

  async removeSaved(jobId: string, snapshotId: string): Promise<void> {
    await request("DELETE", `/saved-jobs/${encodeURIComponent(snapshotId)}/${encodeURIComponent(jobId)}`);
  },
};
