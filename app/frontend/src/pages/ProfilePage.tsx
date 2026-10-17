import { api } from "../lib/api";
import { listSourceProfiles, profileFromSource } from "../lib/importProfile";
import { useRef, useState, type ReactNode } from "react";
import type { EntryKind, PreferredArrangement, Profile, ResumeReader } from "../types";
import {
  ARRANGEMENT_OPTIONS, ENTRY_KINDS, LIMITS, fromDraft, newId, toDraft, validateDraft, type ProfileDraft,
} from "../lib/profile";
import ProfilePreview from "../components/ProfilePreview";
import MlNote from "../components/MlNote";
import { blankProfile, isBlankProfile } from "../lib/profile";

/** A brand-new profile carries a placeholder name for the server; the form shows the name field empty. */
const draftOf = (p: Profile): ProfileDraft => toDraft(isBlankProfile(p) ? { ...p, name: "" } : p);

interface Props {
  profile: Profile;
  edited: boolean;
  recovered: boolean;
  /** "local" (Jacob's app): the resume is read on this computer, so no consent to send it is needed. */
  resumeReader?: ResumeReader;
  onSave: (p: Profile) => Promise<void>;
  onReset: () => Promise<void>;
}

function Field({ id, label, error, hint, children }: { id: string; label: string; error?: string; hint?: string; children: ReactNode }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {children}
      {hint && !error && <p className="field__hint">{hint}</p>}
      {error && <p className="field__error" id={`${id}-err`}>{error}</p>}
    </div>
  );
}

export default function ProfilePage({ profile, edited, resumeReader = "openai", onSave, onReset }: Props) {
  const localReader = resumeReader === "local";
  const [draft, setDraft] = useState<ProfileDraft>(() => draftOf(profile));
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [status, setStatus] = useState<"idle" | "saved" | "cancelled" | "reset">("idle");
  const [confirmReset, setConfirmReset] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [imported, setImported] = useState<{ id: string; profile: ReturnType<typeof listSourceProfiles>[number]["profile"] }[]>([]);
  const [importPick, setImportPick] = useState(0);
  const [importError, setImportError] = useState<string | null>(null);

  const onImportFile = async (file: File | undefined) => {
    setImportError(null);
    setImported([]);
    if (!file) return;
    try {
      const rows = listSourceProfiles(JSON.parse(await file.text()));
      if (rows.length === 0) throw new Error("No profiles found in that file.");
      setImported(rows);
      setImportPick(0);
    } catch (err) {
      setImportError(err instanceof Error ? err.message : "That file could not be read as a profile.");
    }
  };
  const loadImported = () => {
    const row = imported[importPick];
    if (!row) return;
    setDraft(toDraft(profileFromSource(row.profile)));
    setErrors({});
    setStatus("idle");
    setImported([]);
  };
  const [resume, setResume] = useState<{ name: string; source: { text: string } | { pdfBase64: string } } | null>(null);
  const [resumeConsent, setResumeConsent] = useState(false);
  const [filling, setFilling] = useState(false);
  const [fillMessage, setFillMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);

  const onResumeFile = async (file: File | undefined) => {
    setFillMessage(null);
    setResume(null);
    if (!file) return;
    const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
    if (file.size > (isPdf ? 2_000_000 : 200_000)) {
      setFillMessage({ kind: "error", text: isPdf ? "That PDF is too large for a resume (limit 2 MB)." : "That file is too large for a resume. Use a text file under 200 KB." });
      return;
    }
    try {
      if (isPdf) {
        const bytes = new Uint8Array(await file.arrayBuffer());
        let binary = "";
        for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
        setResume({ name: file.name, source: { pdfBase64: btoa(binary) } });
      } else {
        setResume({ name: file.name, source: { text: await file.text() } });
      }
    } catch {
      setFillMessage({ kind: "error", text: "That file could not be read." });
    }
  };
  const magicFill = async () => {
    if (!resume || !(resumeConsent || localReader)) return;
    setFilling(true);
    setFillMessage(null);
    try {
      const result = await api.extractFromResume(resume.source);
      if (result.state === "unavailable") {
        setFillMessage({ kind: "error", text: "Magic fill is turned off on this server, so nothing was sent. Ask whoever runs the demo to start it without the -NoAI option." });
        return;
      }
      const p = result.profile;
      if (p.experience.length === 0 && p.skills.length === 0) {
        setFillMessage({ kind: "error", text: "Nothing usable could be read from that resume, so the form was not changed." });
        return;
      }
      // The resume replaces name, headline, location, experience and skills outright (blank where it says nothing), so
      // values from the example profile cannot mix in. Desired roles, arrangement and constraints are not in a resume:
      // they are kept if you have edited the profile, and cleared if the form still holds the example's.
      setDraft((d) => ({
        ...d,
        name: p.name,
        headline: p.headline,
        location: p.location,
        experience: p.experience.map((x) => ({ id: newId("exp"), kind: x.kind, title: x.title, organization: x.organization, description: x.description })),
        skills: p.skills.map((s) => ({ id: newId("skill"), name: s.name, evidence: s.evidence })),
        ...(edited ? {} : { desiredRolesText: "", arrangements: [], constraints: "" }),
      }));
      setErrors({});
      setStatus("idle");
      setFillMessage({
        kind: "ok",
        text: `Filled ${p.experience.length} experience ${p.experience.length === 1 ? "entry" : "entries"} and ${p.skills.length} ${p.skills.length === 1 ? "skill" : "skills"} from the resume, replacing the name, headline, location, experience and skills in the form (Cancel undoes it).` +
          (result.dropped > 0 ? ` ${result.dropped} suggested ${result.dropped === 1 ? "item was" : "items were"} left out because ${result.dropped === 1 ? "it" : "they"} could not be matched to your resume text.` : "") +
          " Review everything, add your desired roles (a resume does not say what you want next), then press Save.",
      });
    } catch (err) {
      setFillMessage({ kind: "error", text: err instanceof Error ? err.message : "Magic fill failed. Nothing was changed." });
    } finally {
      setFilling(false);
    }
  };
  const summaryRef = useRef<HTMLDivElement>(null);

  const dirty = JSON.stringify(draft) !== JSON.stringify(draftOf(profile));
  const set = <K extends keyof ProfileDraft>(k: K, v: ProfileDraft[K]) => {
    setDraft((d) => ({ ...d, [k]: v }));
    setStatus("idle");
  };

  const err = (k: string) => errors[k];
  const props = (id: string, key: string) => ({
    id,
    "aria-invalid": err(key) ? true : undefined,
    "aria-describedby": err(key) ? `${id}-err` : undefined,
  });

  const save = async () => {
    const e = validateDraft(draft);
    setErrors(e);
    if (Object.keys(e).length) {
      setStatus("idle");
      setTimeout(() => summaryRef.current?.focus(), 0);
      return;
    }
    const next = fromDraft(draft);
    try {
      await onSave(next);
      setSaveError(null);
      setDraft(toDraft(next));
      setStatus("saved");
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Could not save the profile.");
    }
  };

  const cancel = () => {
    setDraft(draftOf(profile));
    setErrors({});
    setStatus("cancelled");
  };

  const toggleArrangement = (a: PreferredArrangement) =>
    set("arrangements", draft.arrangements.includes(a) ? draft.arrangements.filter((x) => x !== a) : [...draft.arrangements, a]);

  const errorList = Object.values(errors);

  return (
    <>
      <main className="browse" aria-label="My profile">
        <header className="browse__head">
          <div>
            <h1>My Profile</h1>
            <p className="browse__sub">Tell us what you have done, what you can do, and what you want next.</p>
          </div>
          <span className="demo-tag">Saved to your demo session</span>
        </header>

        {saveError && (
          <p className="notice notice--compact" role="alert">{saveError}</p>
        )}

        <div className="formscroll">
          <form className="form" onSubmit={(e) => { e.preventDefault(); save(); }} noValidate>
            {errorList.length > 0 && (
              <div className="errorsummary" role="alert" tabIndex={-1} ref={summaryRef}>
                <strong>Fix {errorList.length} {errorList.length === 1 ? "problem" : "problems"} before saving:</strong>
                <ul>{errorList.map((m, i) => <li key={i}>{m}</li>)}</ul>
              </div>
            )}

            <fieldset className="formcard">
              <legend>Import (optional)</legend>
              <p className="section__note">
                Load a profile from a JSON file on this computer (the project's candidate-profile format). It fills the form below so you can
                review it; nothing is saved until you press Save. The file is read in your browser and is not uploaded anywhere else.
              </p>
              <input type="file" accept="application/json,.json" onChange={(e) => void onImportFile(e.target.files?.[0])} aria-label="Profile JSON file" />
              {importError && <p className="field__error" role="alert">{importError}</p>}
              {imported.length > 0 && (
                <p>
                  <select value={importPick} onChange={(e) => setImportPick(Number(e.target.value))} aria-label="Which profile to load">
                    {imported.map((r, i) => <option key={r.id} value={i}>{r.id}</option>)}
                  </select>{" "}
                  <button type="button" className="btn btn--sm" onClick={loadImported}>Load into form</button>
                </p>
              )}
            </fieldset>

            <fieldset className="formcard">
              <legend>Magic fill from a resume (optional)</legend>
              <p className="section__note">
                Choose a text-based PDF or a plain-text (.txt) resume and the form fills in from it. Only what the resume actually says is used;
                anything it cannot find is left as it was. It does not guess your desired roles. Nothing is saved until you press Save.
              </p>
              <MlNote id="magic-fill" variant={resumeReader === "local" || resumeReader === "openai" ? resumeReader : undefined} />
              <input type="file" accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown" onChange={(e) => void onResumeFile(e.target.files?.[0])} aria-label="Resume file (PDF or text)" />
              {localReader ? (
                <p className="muted">
                  The resume is read on this computer: nothing is sent anywhere, and the text is not stored by this app. It reads ordinary
                  one-column resumes best; check every entry before you save. Scanned (image-only) PDFs cannot be read.
                </p>
              ) : (
                <p>
                  <label>
                    <input type="checkbox" checked={resumeConsent} onChange={(e) => setResumeConsent(e.target.checked)} />{" "}
                    I agree to send the text of this resume to OpenAI so it can be read. Emails, phone numbers and links are removed first, and the
                    text is not stored by this app. Scanned (image-only) PDFs cannot be read.
                  </label>
                </p>
              )}
              <p>
                <button type="button" className="btn btn--sm" disabled={!resume || !(resumeConsent || localReader) || filling} onClick={() => void magicFill()}>
                  {filling ? "Reading resume…" : "Magic fill"}
                </button>
                {resume && <span className="muted"> {resume.name}</span>}
              </p>
              {fillMessage && <p className={fillMessage.kind === "error" ? "field__error" : "muted"} role={fillMessage.kind === "error" ? "alert" : "status"}>{fillMessage.text}</p>}
            </fieldset>

            <fieldset className="formcard">
              <legend>About you</legend>
              <Field id="pf-name" label="Name" error={err("name")}>
                <input type="text" value={draft.name} maxLength={LIMITS.name + 20} onChange={(e) => set("name", e.target.value)} {...props("pf-name", "name")} />
              </Field>
              <Field id="pf-headline" label="Headline (optional)" error={err("headline")}>
                <input type="text" value={draft.headline} onChange={(e) => set("headline", e.target.value)} {...props("pf-headline", "headline")} />
              </Field>
            </fieldset>

            <fieldset className="formcard">
              <legend>Experience, projects and education</legend>
              {draft.experience.length === 0 && <p className="muted">No entries yet.</p>}
              {draft.experience.map((x, i) => {
                const patch = (p: Partial<typeof x>) => set("experience", draft.experience.map((y) => (y.id === x.id ? { ...y, ...p } : y)));
                const kind = ENTRY_KINDS.find((k) => k.value === x.kind) ?? ENTRY_KINDS[0];
                return (
                  <div className="entry" key={x.id}>
                    <Field id={`exp-${x.id}-kind`} label={`Type (entry ${i + 1})`}>
                      <select id={`exp-${x.id}-kind`} value={x.kind} onChange={(e) => patch({ kind: e.target.value as EntryKind })}>
                        {ENTRY_KINDS.map((k) => <option key={k.value} value={k.value}>{k.label}</option>)}
                      </select>
                    </Field>
                    <div className="entry__grid">
                      <Field id={`exp-${x.id}-title`} label={kind.titleLabel} error={err(`exp.${x.id}.title`)}>
                        <input type="text" value={x.title} onChange={(e) => patch({ title: e.target.value })} {...props(`exp-${x.id}-title`, `exp.${x.id}.title`)} />
                      </Field>
                      <Field id={`exp-${x.id}-org`} label={kind.orgLabel} error={err(`exp.${x.id}.organization`)}>
                        <input type="text" value={x.organization} onChange={(e) => patch({ organization: e.target.value })} {...props(`exp-${x.id}-org`, `exp.${x.id}.organization`)} />
                      </Field>
                    </div>
                    <Field id={`exp-${x.id}-desc`} label="What you did (optional)" error={err(`exp.${x.id}.description`)} hint={`${x.description.length}/${LIMITS.experienceText}`}>
                      <textarea rows={3} value={x.description} onChange={(e) => patch({ description: e.target.value })} {...props(`exp-${x.id}-desc`, `exp.${x.id}.description`)} />
                    </Field>
                    <button type="button" className="btn btn--ghost btn--sm" onClick={() => set("experience", draft.experience.filter((y) => y.id !== x.id))}>
                      Remove entry {i + 1}
                    </button>
                  </div>
                );
              })}
              <button type="button" className="btn" disabled={draft.experience.length >= LIMITS.maxExperience} onClick={() => set("experience", [...draft.experience, { id: newId("exp"), kind: "job", title: "", organization: "", description: "" }])}>
                Add entry
              </button>
            </fieldset>

            <fieldset className="formcard">
              <legend>Skills and training</legend>
              <p className="field__hint">Add evidence where you can; matches are only as specific as the evidence provided.</p>
              {draft.skills.length === 0 && <p className="muted">No skills yet.</p>}
              {draft.skills.map((s, i) => {
                const patch = (p: Partial<typeof s>) => set("skills", draft.skills.map((y) => (y.id === s.id ? { ...y, ...p } : y)));
                return (
                  <div className="entry" key={s.id}>
                    <div className="entry__grid">
                      <Field id={`sk-${s.id}-name`} label={`Skill or training (entry ${i + 1})`} error={err(`skill.${s.id}.name`)}>
                        <input type="text" value={s.name} onChange={(e) => patch({ name: e.target.value })} {...props(`sk-${s.id}-name`, `skill.${s.id}.name`)} />
                      </Field>
                      <Field id={`sk-${s.id}-ev`} label="Evidence (optional)" error={err(`skill.${s.id}.evidence`)}>
                        <input type="text" value={s.evidence} onChange={(e) => patch({ evidence: e.target.value })} {...props(`sk-${s.id}-ev`, `skill.${s.id}.evidence`)} />
                      </Field>
                    </div>
                    <button type="button" className="btn btn--ghost btn--sm" onClick={() => set("skills", draft.skills.filter((y) => y.id !== s.id))}>
                      Remove skill {i + 1}
                    </button>
                  </div>
                );
              })}
              <button type="button" className="btn" disabled={draft.skills.length >= LIMITS.maxSkills} onClick={() => set("skills", [...draft.skills, { id: newId("skill"), name: "", evidence: "" }])}>
                Add skill
              </button>
            </fieldset>

            <fieldset className="formcard">
              <legend>Job preferences</legend>
              <Field id="pf-roles" label="Desired roles" error={err("roles")} hint="Separate with commas.">
                <input type="text" value={draft.desiredRolesText} onChange={(e) => set("desiredRolesText", e.target.value)} {...props("pf-roles", "roles")} />
              </Field>
              <Field id="pf-location" label="Preferred location (optional)" error={err("location")}>
                <input type="text" value={draft.location} onChange={(e) => set("location", e.target.value)} {...props("pf-location", "location")} />
              </Field>
              <div className="field" role="group" aria-labelledby="pf-arr">
                <span id="pf-arr" className="field__label">Work arrangement</span>
                <div className="checks">
                  {ARRANGEMENT_OPTIONS.map((a) => (
                    <label key={a} className="check">
                      <input type="checkbox" checked={draft.arrangements.includes(a)} onChange={() => toggleArrangement(a)} />
                      {a}
                    </label>
                  ))}
                </div>
              </div>
              <Field id="pf-constraints" label="Other constraints (optional)" error={err("constraints")} hint={`${draft.constraints.length}/${LIMITS.constraints}`}>
                <textarea rows={2} value={draft.constraints} onChange={(e) => set("constraints", e.target.value)} {...props("pf-constraints", "constraints")} />
              </Field>
            </fieldset>

            <div className="formbar">
              <div className="formbar__actions">
                <button type="submit" className="btn btn--primary" disabled={!dirty}>Save profile</button>
                <button type="button" className="btn" disabled={!dirty} onClick={cancel}>Cancel</button>
              </div>
              <p className="formbar__status" role="status" aria-live="polite">
                {dirty ? "Unsaved changes" : status === "saved" ? "Profile saved. Recommendations are refreshing." : status === "cancelled" ? "Edits discarded." : status === "reset" ? "Started over with a blank profile." : ""}
              </p>
              <div className="formbar__reset">
                {confirmReset ? (
                  <>
                    <span className="small">Clear your profile, saved jobs and ratings and start over?</span>
                    <button type="button" className="btn btn--sm" onClick={() => { void onReset().then(() => { setDraft(draftOf(blankProfile)); setErrors({}); setResume(null); setResumeConsent(false); setFillMessage(null); setStatus("reset"); setConfirmReset(false); }).catch((err) => setSaveError(err instanceof Error ? err.message : "Could not reset.")); }}>Yes, reset</button>
                    <button type="button" className="btn btn--ghost btn--sm" onClick={() => setConfirmReset(false)}>Keep mine</button>
                  </>
                ) : (
                  <button type="button" className="btn btn--ghost btn--sm" onClick={() => setConfirmReset(true)}>Start over (blank profile)</button>
                )}
              </div>
            </div>
          </form>

          <div className="preview--inline">
            <ProfilePreview profile={profile} edited={edited} />
          </div>
        </div>
      </main>

      <aside className="panel panel--preview" aria-label="Saved profile preview">
        <ProfilePreview profile={profile} edited={edited} />
      </aside>
    </>
  );
}
