import type { EntryKind, Profile } from "../types";

const GROUPS: { kind: EntryKind; heading: string }[] = [
  { kind: "job", heading: "Experience" },
  { kind: "project", heading: "Projects" },
  { kind: "education", heading: "Education" },
];

export default function ProfilePreview({ profile, edited }: { profile: Profile; edited: boolean }) {
  if (!edited) {
    // Nothing saved yet: do not show the fictional starting profile's details as if they were the user's.
    return (
      <div className="preview">
        <h2 className="preview__title">Your saved profile</h2>
        <div className="preview__empty">
          <strong>No profile saved yet</strong>
          <p>
            Fill in the form or use Magic fill, then press <strong>Save profile</strong>. Your saved profile will appear here.
          </p>
          <p className="muted">Until you save one, Discover has nothing to match against.</p>
        </div>
      </div>
    );
  }
  return (
    <div className="preview">
      <h2 className="preview__title">Saved profile</h2>
      <p className="section__note">
        {edited ? "Your edited profile." : "Starting example profile; edit it to match your own background."}{" "}
        Discover ranks listings against this profile. Anything not written here counts as unknown, not as missing.
      </p>

      <div className="preview__id">
        <strong>{profile.name}</strong>
        {profile.headline && <span className="muted">{profile.headline}</span>}
      </div>

      {profile.experience.length === 0 && (
        <section className="section">
          <h3>Experience</h3>
          <p className="muted">None added.</p>
        </section>
      )}
      {GROUPS.map(({ kind, heading }) => {
        const entries = profile.experience.filter((x) => x.kind === kind);
        return entries.length === 0 ? null : (
          <section className="section" key={kind}>
            <h3>{heading}</h3>
            <ul className="preview__list">
              {entries.map((x) => (
                <li key={x.id}>
                  <strong>{x.title}</strong>{x.organization && <span className="muted"> · {x.organization}</span>}
                  {x.description && <p>{x.description}</p>}
                </li>
              ))}
            </ul>
          </section>
        );
      })}

      <section className="section">
        <h3>Skills and training</h3>
        {profile.skills.length === 0 ? <p className="muted">None added.</p> : (
          <ul className="preview__list">
            {profile.skills.map((s) => (
              <li key={s.id}>
                <strong>{s.name}</strong>
                {s.evidence ? <p>{s.evidence}</p> : <p className="muted">No evidence added.</p>}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="section">
        <h3>Preferences</h3>
        <dl className="kv kv--wide">
          <div><dt>Desired roles</dt><dd className={profile.desiredRoles.length ? "" : "muted"}>{profile.desiredRoles.join(", ") || "Not provided"}</dd></div>
          <div><dt>Location</dt><dd className={profile.location ? "" : "muted"}>{profile.location || "Not provided"}</dd></div>
          <div><dt>Work arrangement</dt><dd className={profile.arrangements.length ? "" : "muted"}>{profile.arrangements.join(", ") || "No preference"}</dd></div>
          <div><dt>Other constraints</dt><dd className={profile.constraints ? "" : "muted"}>{profile.constraints || "None"}</dd></div>
        </dl>
      </section>
    </div>
  );
}
