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
  /** ISO date (YYYY-MM-DD) of the snapshot the listing comes from (the same for every listing). */
  observedOn: string;
  /** ISO date the listing was listed: the employer's posted date, or the crawler's first sighting when that is earlier
   *  or no posted date is given (listedBasis says which). Null when neither is known. */
  listedOn: string | null;
  listedBasis: "posted" | "first_seen" | "unknown";
  historicalStatus: HistoricalStatus;
  level?: JobLevel;
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
  /** Set when the listing was added to test the matching (the two methods disagreed about it, or it was random). */
  exploration?: { source: "keyword" | "embedding" | "random" } | null;
}

export type PreferredArrangement = Exclude<WorkArrangement, "Not provided">;

/** How Magic fill reads a resume on this server: sent to OpenAI, read on this computer (Jacob's app), or switched off. */
export type ResumeReader = "openai" | "local" | "off";

/** The days the "Open on" picker offers: the first listing day to the last sighting in the snapshot. */
export interface ListedRange { first: string | null; last: string | null }

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

/** The listing's level, from PredictLeads' seniority plus the title (backend `job_level`). */
export type JobLevel = "internship" | "program" | "gig" | "entry" | "experienced" | "senior" | "manager" | "director" | "unstated";

export const JOB_LEVELS: { id: JobLevel; label: string }[] = [
  { id: "internship", label: "Internship" },
  { id: "program", label: "Program (graduate, apprenticeship, fellowship)" },
  { id: "gig", label: "Gig / AI data work" },
  { id: "entry", label: "Entry level" },
  { id: "experienced", label: "Experienced" },
  { id: "senior", label: "Senior" },
  { id: "manager", label: "Manager" },
  { id: "director", label: "Director and above" },
  { id: "unstated", label: "Level not stated" },
];

export interface Filters {
  query: string;
  location: string;
  description: DescriptionFilter;
  /** Levels to show; empty shows every level. */
  levels: JobLevel[];
  /** Show only the listings open on this day (YYYY-MM-DD); empty shows every listing. */
  asOf?: string;
}

export type FeedbackSignal = "good" | "bad";
export type FeedbackReason = "wrong_role" | "seniority" | "location" | "pay" | "company" | "seen" | "other";

/** How the last search matched: no query, in the job title, or (when few titles match) anywhere in the listing. */
export type SearchScope = "all" | "title" | "anywhere";

/* ---------- Compass ---------- */

export type RequirementStatus = "supported" | "partial" | "unknown";

export interface CompassRole {
  name: string;
  /** Compass heading in degrees, clockwise from north. */
  angle: number;
  /** Jobs on the map in this role. */
  count: number;
}

/** Where the person's scope points, and how wide it is when each map heading faces north. */
export interface CompassScope {
  /** Map heading (degrees) with the most well-fitting jobs around it; the bezel opens with this facing north. */
  heading: number;
  /** Degrees between entries of halfWidths. */
  step: number;
  /** Scope half-width at map headings 0, step, 2*step, ... */
  halfWidths: number[];
}

export interface CompassTheme {
  id: string;
  name: string;
  /** Number of picked jobs in this theme. */
  count: number;
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

/** One spoke of a job's strengths radar: a broad area of the listing's requirement lines. */
export interface CompassAreaScore {
  id: string;
  name: string;
  /** How much of the listing's requirement text is in this area; the most-asked scored area is 100. */
  asked: number;
  /** Requirement lines in this area. */
  lines: number;
  /** 0-100 closeness of the profile's lines to this area's requirements; null with no lines or for credentials, which similarity can't judge. */
  shown: number | null;
}

/** Credentials & eligibility: never scored, so it gets no spoke. */
export const CREDENTIALS_AREA = "credentials";

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
  /** Role-family spoke this job sits on. */
  role: string;
  /** Compass heading in degrees, clockwise from north, inside its role's slice. */
  angle: number;
  why: CompassWhy[];
  /** Requirement-group ids this pick asks for that the profile does not fully show. */
  stretch: string[];
  /** For suggestions: requirement groups the job would help build. */
  builds: string[];
  /** This job's requirement lines by skill area. */
  areas: CompassAreaScore[];
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
  /** Up to 12 role-family spokes around the map, evenly spaced. */
  roles: CompassRole[];
  scope: CompassScope;
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
