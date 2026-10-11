import type { FeedbackReason } from "../types";

export const NOTE_MAX = 200;

/** Why a job was not for you. Only the first and the last push results away from similar jobs; the rest only hide that job. */
export const REASONS: { id: FeedbackReason; label: string }[] = [
  { id: "wrong_role", label: "Wrong kind of role" },
  { id: "seniority", label: "Level doesn't fit" },
  { id: "location", label: "Location or remote" },
  { id: "pay", label: "Pay" },
  { id: "company", label: "Not this company" },
  { id: "seen", label: "Already seen it" },
  { id: "other", label: "Something else" },
];
