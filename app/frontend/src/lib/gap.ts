import type { Profile } from "../types";

export type RequirementKind = "required" | "preferred" | "mentioned";

export interface RequirementLine {
  text: string;
  kind: RequirementKind;
  /** Profile skills whose names appear in this line (a keyword match, not an assessment). */
  matchedSkills: string[];
}

const REQUIRED = /\b(required|requirements?|must|minimum|at least|\d+\+?\s*years?)\b/i;
const PREFERRED = /\b(preferred|nice to have|bonus|a plus|desired|ideal(ly)?)\b/i;
const SKILL_CUE = /\b(experience (with|in|using)|proficien\w*|knowledge of|familiar\w*|degree|certif\w*|skills?|expertise|background in|ability to)\b/i;
const NEGATED = /\b(not required|no\s+[\w-]+\s+(is\s+)?required|not necessary|not mandatory)\b/i;

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function skillMatches(skill: string, line: string): boolean {
  const name = skill.trim().toLowerCase();
  if (!name) return false;
  const hay = line.toLowerCase();
  if (new RegExp(`(^|[^a-z0-9+#])${escapeRegExp(name)}([^a-z0-9+#]|$)`).test(hay)) return true;
  // Multi-word skill: every meaningful word must appear.
  const words = name.split(/[^a-z0-9+#]+/).filter((w) => w.length >= 4);
  return words.length >= 2 && words.every((w) => hay.includes(w));
}

/** Pulls requirement-like lines out of a plain-text description. Pure keyword rules; nothing is inferred. */
export function requirementLines(description: string | null, profile: Profile, max = 14): RequirementLine[] {
  if (!description) return [];
  const seen = new Set<string>();
  const out: RequirementLine[] = [];
  const candidates = description
    .split(/\n+/)
    .flatMap((line) => (line.length > 260 ? line.split(/(?<=[.!?])\s+/) : [line]))
    .map((l) => l.replace(/^[-*•]\s*/, "").trim())
    .filter((l) => l.length >= 12 && l.length <= 260);

  for (const text of candidates) {
    const key = text.toLowerCase();
    if (seen.has(key)) continue;
    const negated = NEGATED.test(text);
    const required = REQUIRED.test(text) && !negated;
    const preferred = PREFERRED.test(text);
    const cue = SKILL_CUE.test(text);
    if (!required && !preferred && !cue) continue;
    seen.add(key);
    out.push({
      text,
      kind: preferred && !required ? "preferred" : required ? "required" : "mentioned",
      matchedSkills: profile.skills.filter((s) => skillMatches(s.name, text)).map((s) => s.name),
    });
    if (out.length >= max) break;
  }
  return out;
}
