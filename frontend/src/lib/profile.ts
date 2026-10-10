import type { EntryKind, Profile, PreferredArrangement } from "../types";

export const LIMITS = {
  name: 80,
  headline: 120,
  title: 120,
  organization: 80,
  experienceText: 1200,
  skillName: 60,
  evidence: 400,
  role: 60,
  location: 80,
  constraints: 300,
  maxExperience: 15,
  maxSkills: 40,
  maxRoles: 8,
} as const;

/** The backend needs a non-empty name, so a brand-new profile carries this placeholder; the form shows the name field empty. */
export const BLANK_NAME = "New profile";

export const blankProfile: Profile = {
  name: BLANK_NAME,
  headline: "",
  experience: [],
  skills: [],
  desiredRoles: [],
  location: "",
  arrangements: [],
  constraints: "",
};

/** True for a profile nobody has filled in yet (nothing to match against). */
export function isBlankProfile(p: Profile): boolean {
  return (
    p.name === BLANK_NAME && !p.headline && !p.location && !p.constraints &&
    p.experience.length === 0 && p.skills.length === 0 && p.desiredRoles.length === 0 && p.arrangements.length === 0
  );
}

export const ENTRY_KINDS: { value: EntryKind; label: string; titleLabel: string; orgLabel: string }[] = [
  { value: "job", label: "Job", titleLabel: "Role title", orgLabel: "Organization (optional)" },
  { value: "project", label: "Project", titleLabel: "Project name", orgLabel: "Where or for whom (optional)" },
  { value: "education", label: "Education", titleLabel: "Degree or program", orgLabel: "School (optional)" },
];

export const isEntryKind = (v: unknown): v is EntryKind => ENTRY_KINDS.some((k) => k.value === v);

export const ARRANGEMENT_OPTIONS: PreferredArrangement[] = ["Remote", "Hybrid", "On-site"];

/** Form-friendly shape: desired roles are edited as comma-separated text. */
export interface ProfileDraft extends Omit<Profile, "desiredRoles"> {
  desiredRolesText: string;
}

export function toDraft(p: Profile): ProfileDraft {
  const { desiredRoles, ...rest } = p;
  return { ...rest, desiredRolesText: desiredRoles.join(", ") };
}

export function parseRoles(text: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const part of text.split(",")) {
    const t = part.trim();
    if (t && !seen.has(t.toLowerCase())) {
      seen.add(t.toLowerCase());
      out.push(t);
    }
  }
  return out;
}

export function fromDraft(d: ProfileDraft): Profile {
  const { desiredRolesText, ...rest } = d;
  return {
    ...rest,
    name: rest.name.trim(),
    headline: rest.headline.trim(),
    location: rest.location.trim(),
    constraints: rest.constraints.trim(),
    experience: rest.experience.map((e) => ({
      ...e,
      title: e.title.trim(),
      organization: e.organization.trim(),
      description: e.description.trim(),
    })),
    skills: rest.skills.map((s) => ({ ...s, name: s.name.trim(), evidence: s.evidence.trim() })),
    desiredRoles: parseRoles(desiredRolesText),
  };
}

/** Flat map of field key to message; empty when valid. */
export function validateDraft(d: ProfileDraft): Record<string, string> {
  const e: Record<string, string> = {};
  if (!d.name.trim()) e.name = "Enter a name.";
  else if (d.name.trim().length > LIMITS.name) e.name = `Use ${LIMITS.name} characters or fewer.`;
  if (d.headline.length > LIMITS.headline) e.headline = `Use ${LIMITS.headline} characters or fewer.`;

  d.experience.forEach((x) => {
    if (!x.title.trim()) e[`exp.${x.id}.title`] = "Enter a role title, or remove this entry.";
    else if (x.title.length > LIMITS.title) e[`exp.${x.id}.title`] = `Use ${LIMITS.title} characters or fewer.`;
    if (x.organization.length > LIMITS.organization) e[`exp.${x.id}.organization`] = `Use ${LIMITS.organization} characters or fewer.`;
    if (x.description.length > LIMITS.experienceText) e[`exp.${x.id}.description`] = `Use ${LIMITS.experienceText} characters or fewer.`;
  });

  const seen = new Set<string>();
  d.skills.forEach((s) => {
    const n = s.name.trim().toLowerCase();
    if (!n) e[`skill.${s.id}.name`] = "Enter a skill name, or remove this entry.";
    else if (n.length > LIMITS.skillName) e[`skill.${s.id}.name`] = `Use ${LIMITS.skillName} characters or fewer.`;
    else if (seen.has(n)) e[`skill.${s.id}.name`] = "This skill is already listed.";
    seen.add(n);
    if (s.evidence.length > LIMITS.evidence) e[`skill.${s.id}.evidence`] = `Use ${LIMITS.evidence} characters or fewer.`;
  });

  const roles = parseRoles(d.desiredRolesText);
  if (roles.length > LIMITS.maxRoles) e.roles = `List at most ${LIMITS.maxRoles} roles.`;
  else if (roles.some((r) => r.length > LIMITS.role)) e.roles = `Keep each role under ${LIMITS.role} characters.`;
  if (d.location.length > LIMITS.location) e.location = `Use ${LIMITS.location} characters or fewer.`;
  if (d.constraints.length > LIMITS.constraints) e.constraints = `Use ${LIMITS.constraints} characters or fewer.`;
  return e;
}

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** Validates untrusted stored data; returns null if it cannot be used as-is. */
export function validateStoredProfile(v: unknown): Profile | null {
  if (!isObj(v)) return null;
  const { name, headline, experience, skills, desiredRoles, location, arrangements, constraints } = v;
  if (typeof name !== "string" || typeof headline !== "string" || typeof location !== "string" || typeof constraints !== "string") return null;
  if (!Array.isArray(experience) || !Array.isArray(skills) || !Array.isArray(desiredRoles) || !Array.isArray(arrangements)) return null;

  const exp = [];
  for (const x of experience) {
    if (!isObj(x) || typeof x.id !== "string" || typeof x.title !== "string" || typeof x.organization !== "string" || typeof x.description !== "string") return null;
    exp.push({ id: x.id, kind: isEntryKind(x.kind) ? x.kind : "job", title: x.title, organization: x.organization, description: x.description });
  }
  const sk = [];
  for (const x of skills) {
    if (!isObj(x) || typeof x.id !== "string" || typeof x.name !== "string" || typeof x.evidence !== "string") return null;
    sk.push({ id: x.id, name: x.name, evidence: x.evidence });
  }
  if (!desiredRoles.every((r) => typeof r === "string")) return null;
  if (!arrangements.every((a) => (ARRANGEMENT_OPTIONS as unknown[]).includes(a))) return null;

  const profile: Profile = {
    name, headline, location, constraints,
    experience: exp, skills: sk,
    desiredRoles: desiredRoles as string[],
    arrangements: arrangements as PreferredArrangement[],
  };
  return Object.keys(validateDraft(toDraft(profile))).length === 0 ? profile : null;
}

export function newId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 9)}`;
}
