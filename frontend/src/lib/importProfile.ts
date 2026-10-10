import type { PreferredArrangement, Profile } from "../types";
import { LIMITS, newId } from "./profile";

/** A profile in the project's candidate-profile JSON format (see code/ai_ml_candidate_profiles.json). */
interface SourceProfile {
  profile_id?: string;
  display_name?: string;
  career_transition_goal?: string;
  work_history?: string[];
  accomplishments?: string[];
  skills_with_evidence?: { skill: string; evidence: string }[];
  education_certifications?: string[];
  desired_roles?: string[];
  location?: string | null;
  work_arrangement_preferences?: string[];
}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

const clip = (s: string, n: number) => (s.length <= n ? s : `${s.slice(0, n - 1).trimEnd()}…`);

function prettyName(profileId: string): string {
  return profileId
    .replace(/^(real|ai)_/, "")
    .replace(/_(notitle|untitled)$/, "")
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function arrangementsFrom(values: string[] | undefined): PreferredArrangement[] {
  const text = (values ?? []).join(" ").toLowerCase();
  const out: PreferredArrangement[] = [];
  if (text.includes("remote")) out.push("Remote");
  if (text.includes("hybrid")) out.push("Hybrid");
  if (text.includes("on-site") || text.includes("onsite")) out.push("On-site");
  return out;
}

export function listSourceProfiles(json: unknown): { id: string; profile: SourceProfile }[] {
  const rows = Array.isArray(json) ? json : isObj(json) && Array.isArray(json.profiles) ? json.profiles : isObj(json) ? [json] : [];
  return rows
    .filter((r): r is SourceProfile => isObj(r) && Array.isArray((r as SourceProfile).work_history))
    .map((r, i) => ({ id: r.profile_id ?? `profile ${i + 1}`, profile: r }));
}

/** Converts a source profile into the form's Profile. Nothing is invented; long text is clipped to the form's limits. */
export function profileFromSource(src: SourceProfile): Profile {
  const experience = [
    ...(src.work_history ?? []).map((text) => ({
      id: newId("exp"),
      kind: "job" as const,
      title: clip((text.split(/[:,]/)[0] ?? text).trim() || "Experience", LIMITS.title),
      organization: "",
      description: clip(text, LIMITS.experienceText),
    })),
    ...(src.accomplishments ?? []).map((text) => ({
      id: newId("exp"),
      kind: /project/i.test(text) ? ("project" as const) : ("job" as const),
      title: /project/i.test(text) ? "Project" : "Accomplishment",
      organization: "",
      description: clip(text, LIMITS.experienceText),
    })),
    ...(src.education_certifications ?? []).map((text) => ({
      id: newId("exp"),
      kind: "education" as const,
      title: clip(text.split(",")[0].trim() || "Education", LIMITS.title),
      organization: "",
      description: clip(text, LIMITS.experienceText),
    })),
  ].slice(0, LIMITS.maxExperience);

  const seen = new Set<string>();
  const skills = (src.skills_with_evidence ?? [])
    .filter((s) => {
      const key = s.skill.trim().toLowerCase();
      if (!key || seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .map((s) => ({ id: newId("skill"), name: clip(s.skill.trim(), LIMITS.skillName), evidence: clip(s.evidence ?? "", LIMITS.evidence) }))
    .slice(0, LIMITS.maxSkills);

  return {
    name: clip(src.display_name?.trim() || prettyName(src.profile_id ?? "Imported profile"), LIMITS.name),
    headline: clip(src.career_transition_goal ?? "", LIMITS.headline),
    experience,
    skills,
    desiredRoles: (src.desired_roles ?? []).map((r) => clip(r.trim(), LIMITS.role)).filter(Boolean).slice(0, LIMITS.maxRoles),
    location: clip(src.location ?? "", LIMITS.location),
    arrangements: arrangementsFrom(src.work_arrangement_preferences),
    constraints: "",
  };
}
