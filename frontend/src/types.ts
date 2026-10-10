export type WorkArrangement = "Remote" | "Hybrid" | "On-site" | "Not provided";

export type HistoricalStatus = "listed_when_observed" | "not_in_latest_snapshot" | "unknown_status";

/** Every field except `name` may be unknown in source data. */
export interface Company {
  /** PredictLeads company id; lets a job be matched to the hiring signals. */
  id?: string | null;
  name: string;
  domain: string | null;
  /** Description as provided by the source data. */
  description: string | null;
  industry: string | null;
  size: string | null;
  headquarters: string | null;
}

export interface Job {
  id: string;
  title: string;
  company: Company;
  location: string | null;
  workArrangement: WorkArrangement;
  salary: string | null;
  /** Plain text. Blank line separates paragraphs; lines starting "- " are bullets. */
  description: string | null;
  sourceUrl: string;
  /** ISO date (YYYY-MM-DD) on which the listing was observed. */
  observedOn: string;
  historicalStatus: HistoricalStatus;
}

/** Present when the backend tied this result to a job the user marked as a good match. */
export interface SimilarToLiked {
  jobId: string;
  title: string | null;
  similarity: number;
}

/** One ranked result as the matching backend returns it. Rank is a retrieval order, not a match percentage. */
export interface Recommendation {
  job: Job;
  rank?: number;
  similarTo?: SimilarToLiked | null;
}

export type PreferredArrangement = Exclude<WorkArrangement, "Not provided">;

export type EntryKind = "job" | "project" | "education";

export interface ExperienceEntry {
  id: string;
  /** How the entry is grouped on screen; matching reads every entry the same way. */
  kind: EntryKind;
  title: string;
  organization: string;
  description: string;
}

export interface SkillEntry {
  id: string;
  name: string;
  evidence: string;
}

export interface Profile {
  name: string;
  headline: string;
  experience: ExperienceEntry[];
  skills: SkillEntry[];
  desiredRoles: string[];
  location: string;
  arrangements: PreferredArrangement[];
  constraints: string;
}

/** User-chosen workflow state. Unrelated to the source listing's status. */
export const STAGES = ["Saved", "Considering", "Applied", "Interviewing", "Archived"] as const;
export type Stage = (typeof STAGES)[number];

export interface TrackedJob {
  stage: Stage;
  note: string;
  /** ISO timestamp. */
  savedAt: string;
}

/** Keyed by stable job ID. */
export type Tracker = Record<string, TrackedJob>;

export type DescriptionFilter = "any" | "with" | "without";

export interface Filters {
  query: string;
  location: string;
  description: DescriptionFilter;
}

export type FeedbackSignal = "good" | "bad";

/** How the last search matched: no query, in the job title, or (when few titles match) anywhere in the listing. */
export type SearchScope = "all" | "title" | "anywhere";

/* ---------- Compass ---------- */

export type RequirementStatus = "supported" | "partial" | "unknown";

export interface CompassTheme {
  id: string;
  name: string;
  /** Number of picked jobs in this theme. */
  count: number;
  /** Compass heading in degrees, clockwise from north. */
  angle: number;
  halfWidth: number;
  /** Mean fit (0-100) of the picks in this theme. */
  meanFit: number;
}

/** A verbatim line of the person's own profile matched to a requirement line. */
export interface CompassEvidence {
  text: string;
  source: string;
  matchedRequirement: string;
}

export interface CompassRequirement {
  id: string;
  /** A requirement sentence as written in the listings (the group's most typical wording). */
  text: string;
  /** How many of the picked jobs ask for this. */
  n: number;
  /** 0-100 similarity of the closest profile line; not a probability. */
  coverage: number;
  status: RequirementStatus;
  evidence: CompassEvidence | null;
  /** Mentions years, a degree or a certification: similarity cannot confirm it, so check it yourself. */
  verify: boolean;
  jobIds: string[];
}

export interface CompassWhy {
  evidence: string;
  source: string;
  requirement: string;
}

export interface CompassJob {
  job: Job;
  pick: boolean;
  themeId: string;
  /** 0-100 similarity to the profile, scaled against all listings. Not a chance of being hired. */
  fit: number;
  /** Degrees from the theme's heading. */
  offset: number;
  why: CompassWhy[];
  /** Requirement-group ids this pick asks for that the profile does not fully show. */
  stretch: string[];
  /** For suggestions: requirement groups the job would help build. */
  builds: string[];
}

export interface CompassStep {
  requirementId: string;
  text: string;
  n: number;
  coverage: number;
  status: RequirementStatus;
  verify: boolean;
  action: string;
}

export interface CompassResult {
  picksAnalyzed: number;
  duplicatesCollapsed: number;
  themes: CompassTheme[];
  strongestThemeId: string;
  jobs: CompassJob[];
  requirements: CompassRequirement[];
  nextSteps: CompassStep[];
  stats: { jobs: number; themes: number; requirementGroups: number; groupsWithEvidence: number; suggestions: number; sentencesRead: number };
  limits: string[];
}

/* ---------- hiring signals ---------- */

export type SignalPattern = "burst" | "sustained" | "cooling" | "quiet";
export type SignalFamily = "ai_ml" | "data" | "technical_sales";

export interface SignalEvidence { job_id: string; title: string; first_seen: string }

export interface SignalCompany {
  company_id: string;
  name: string | null;
  domain: string | null;
  pattern: SignalPattern;
  new_30: number;
  active: number;
  gone_30: number;
  baseline_monthly: number | null;
  surge_z: number | null;
  enough_history: boolean;
  families: Record<SignalFamily, { new_30: number; active: number }>;
  /** New openings per week, oldest first, ending on the snapshot date. */
  weekly: number[];
  intent: { family: string; signal: string; meaning: string; new_30: number | null; baseline_monthly: number | null; top_tech: string[] }[];
  evidence?: SignalEvidence[];
  technologies?: { technology: string; new_30: number; still_observed: number }[];
  week_ends?: string[];
}

export interface SignalsOverview {
  snapshot: string | null;
  companies: number;
  pattern_counts: Record<SignalPattern, number>;
  week_ends: string[];
  carry_over: { pattern: SignalPattern; companies: number; next_as_share_of_this_month: number; share_still_hiring_next_month: number }[];
}

export interface SignalsQuery {
  pattern: SignalPattern | null;
  family: SignalFamily | null;
  q: string;
  sort: "new_30" | "active" | "surge_z";
  limit: number;
  offset: number;
}

/* ---------- AI skills gap analysis (all Compass picks, one request) ---------- */

export type QualificationStatus = "supported" | "explicit_gap" | "unknown";

export interface GapEvidence { evidence_id: string; source: string; text: string }

/** One job's result. `evidence_catalog` holds that job's own lines; the profile's lines come once in the batch. */
export interface GapAssessment {
  evidence_sufficiency: "sufficient" | "insufficient";
  insufficiency_reason: string | null;
  transferable_experience: { job_duty: string; candidate_evidence_ids: string[]; job_evidence_ids: string[]; rationale: string }[];
  qualifications: {
    qualification_id: string; qualification: string; classification: "required" | "preferred" | "unclear";
    status: QualificationStatus; candidate_evidence_ids: string[]; job_evidence_ids: string[]; rationale: string;
  }[];
  next_steps: { action: string; qualification_ids: string[]; evidence_ids: string[] }[];
  evidence_catalog: GapEvidence[];
}

export type GapJobResult = { state: "completed"; assessment: GapAssessment } | { state: "failed"; reason: string };

export type GapBatchResult =
  | { state: "unavailable" }
  | { state: "completed"; candidateEvidence: GapEvidence[]; jobs: Record<string, GapJobResult>; failed: number };
