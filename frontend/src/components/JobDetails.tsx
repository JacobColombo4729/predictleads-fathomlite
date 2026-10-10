import CompanyHiringBadge from "./signals/CompanyHiringBadge";
import CompanyHiring from "./signals/CompanyHiring";
import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import type { Company, Profile, Recommendation, Stage, TrackedJob } from "../types";
import { descriptionParagraphs, formatDate, historicalStatusText, safeHttpUrl } from "../lib/jobs";
import { api } from "../lib/api";
import { requirementLines } from "../lib/gap";
import { NOTE_LIMIT } from "../state/constants";
import { BackIcon, BookmarkIcon, ExternalIcon } from "./Icons";
import StageSelect from "./StageSelect";

type Tab = "overview" | "evidence" | "company";
const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "evidence", label: "Skills check" },
  { id: "company", label: "Company" },
];

interface Props {
  item: Recommendation | null;
  tracked: TrackedJob | undefined;
  profile: Profile;
  snapshotId: string | null;
  onToggleSave: () => void;
  onStageChange: (s: Stage) => void;
  onNoteChange: (note: string) => void;
  onBack: () => void;
  backLabel?: string;
  /** Show the back control on desktop too (used when the panel can be dismissed). */
  backAlways?: boolean;
}

export default function JobDetails(props: Props) {
  const { item } = props;
  if (!item) {
    return (
      <div className="details details--empty">
        <p>Select a job to see its details.</p>
      </div>
    );
  }
  // Keyed so each job opens on Overview.
  return <DetailsBody key={item.job.id} {...props} item={item} />;
}

function DetailsBody({ item, tracked, profile, snapshotId, onToggleSave, onStageChange, onNoteChange, onBack, backLabel = "Back to results", backAlways }: Props & { item: Recommendation }) {
  const [tab, setTab] = useState<Tab>("overview");
  const [fullDescription, setFullDescription] = useState<string | null>(null);
  const [descriptionState, setDescriptionState] = useState<"loading" | "ready" | "failed">("loading");
  const job = item.job;

  useEffect(() => {
    let cancelled = false;
    if (!snapshotId) return;
    api.jobDetails(job.id, snapshotId)
      .then((full) => { if (!cancelled) { setFullDescription(full.description); setDescriptionState("ready"); } })
      .catch(() => { if (!cancelled) setDescriptionState("failed"); });
    return () => { cancelled = true; };
  }, [job.id, snapshotId]);

  const description = fullDescription ?? job.description;
  const blocks = description ? descriptionParagraphs(description) : [];
  const href = safeHttpUrl(job.sourceUrl);
  const saved = Boolean(tracked);
  const lines = useMemo(() => (descriptionState === "ready" ? requirementLines(fullDescription, profile) : []), [descriptionState, fullDescription, profile]);

  const onTabKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = TABS.findIndex((t) => t.id === tab);
    const next = e.key === "ArrowRight" ? (i + 1) % TABS.length : e.key === "ArrowLeft" ? (i + TABS.length - 1) % TABS.length : -1;
    if (next < 0) return;
    e.preventDefault();
    setTab(TABS[next].id);
    document.getElementById(`tab-${TABS[next].id}`)?.focus();
  };

  return (
    <div className="details">
      <button type="button" className={`details__back${backAlways ? " details__back--always" : ""}`} onClick={onBack}>
        <BackIcon />{backLabel}
      </button>

      <header className="details__head">
        <div>
          <h2 className="details__title">{job.title}</h2>
          <p className="details__company">{job.company.name}</p>
          <CompanyHiringBadge companyId={job.company.id} />
        </div>
        <button
          type="button"
          className={`btn btn--ghost${saved ? " btn--saved" : ""}`}
          aria-pressed={saved}
          onClick={onToggleSave}
        >
          <BookmarkIcon filled={saved} />{saved ? "Saved" : "Save"}
        </button>
      </header>

      <div className="details__facts">
        <span>{job.location ?? "Location not provided"}</span>
        <span>Arrangement not provided</span>
        <span>{job.salary ?? "Salary not provided"}</span>
      </div>

      {tracked && (
        <section className="tracking" aria-label="My tracking">
          <div className="tracking__row">
            <label className="tracking__field">
              <span>My stage</span>
              <StageSelect value={tracked.stage} label="My stage" onChange={onStageChange} />
            </label>
            <p className="tracking__hint">Your own workflow state, separate from the listing's status.</p>
          </div>
          <label className="tracking__field">
            <span>Private note</span>
            <textarea
              rows={3}
              maxLength={NOTE_LIMIT}
              value={tracked.note}
              placeholder="Add a note for yourself"
              onChange={(e) => onNoteChange(e.target.value)}
            />
          </label>
          <p className="tracking__hint">Saved to your demo session on this machine.</p>
        </section>
      )}

      <div className="tabs" role="tablist" aria-label="Job sections" onKeyDown={onTabKey}>
        {TABS.map((t) => (
          <button
            key={t.id}
            id={`tab-${t.id}`}
            role="tab"
            type="button"
            className="tabs__tab"
            aria-selected={tab === t.id}
            aria-controls={`panel-${t.id}`}
            tabIndex={tab === t.id ? 0 : -1}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div id={`panel-${tab}`} role="tabpanel" aria-labelledby={`tab-${tab}`} className="tabpanel">
        {tab === "overview" && (
          <>
            <section className="section section--first">
              <h3>Listing</h3>
              <dl className="kv">
                <div><dt>Observed</dt><dd>{formatDate(job.observedOn)}</dd></div>
                <div><dt>Status in snapshot</dt><dd>{historicalStatusText[job.historicalStatus]}</dd></div>
                <div><dt>Current availability</dt><dd className="muted">Not verified. This is a historical listing.</dd></div>
                <div>
                  <dt>Source</dt>
                  <dd>
                    {href ? (
                      <a className="link" href={href} target="_blank" rel="noopener noreferrer">
                        View source listing<ExternalIcon />
                      </a>
                    ) : <span className="muted">No valid link in the source data</span>}
                  </dd>
                </div>
              </dl>
            </section>
            <section className="section">
              <h3>Source description</h3>
              {descriptionState === "loading" && !fullDescription && <p className="muted">Loading the full description…</p>}
              {blocks.length ? (
                <div className="prose">
                  {blocks.map((b, i) =>
                    b.type === "p" ? (
                      <p key={i}>{b.items[0]}</p>
                    ) : (
                      <ul key={i}>{b.items.map((li) => <li key={li}>{li}</li>)}</ul>
                    ),
                  )}
                </div>
              ) : descriptionState !== "loading" ? (
                <p className="notice">
                  No description was captured for this listing, so requirements cannot be compared. Open the source listing to read the full posting.
                </p>
              ) : null}
            </section>
          </>
        )}

        {tab === "evidence" && (
          <section className="section section--first">
            <h3>Skills check</h3>
            <p className="section__note">
              Lines in the listing that mention requirements, next to skills on your profile whose names appear in the same line.
              This is a keyword match, not an assessment. A line with no match means “not evidenced in your profile”, which is
              different from “you don't have this skill”.
            </p>
            {descriptionState === "loading" && <p className="muted">Reading the listing…</p>}
            {descriptionState === "failed" && <p className="notice">Could not load the full listing text.</p>}
            {descriptionState === "ready" && lines.length === 0 && (
              <p className="notice">No requirement-style lines were found in this listing's text.</p>
            )}
            {lines.length > 0 && (
              <ul className="gap">
                {lines.map((l) => (
                  <li key={l.text} className="gap__row">
                    <span className={`gap__kind gap__kind--${l.kind}`}>{l.kind === "required" ? "Required" : l.kind === "preferred" ? "Preferred" : "Mentioned"}</span>
                    <span className="gap__text">{l.text}</span>
                    <span className={l.matchedSkills.length ? "gap__yes" : "gap__unknown"}>
                      {l.matchedSkills.length ? `On your profile: ${l.matchedSkills.join(", ")}` : "Not evidenced in your profile"}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        )}

        {tab === "company" && <CompanyPanel company={job.company} />}
      </div>
    </div>
  );
}

function CompanyPanel({ company }: { company: Company }) {
  const rows: [string, string | null][] = [
    ["What they do", company.description],
    ["Industry", company.industry],
    ["Domain", company.domain],
    ["Size", company.size],
    ["Headquarters", company.headquarters],
  ];
  return (
    <>
    <CompanyHiring companyId={company.id} />
    <section className="section">
      <h3>About {company.name}</h3>
      <p className="section__note">Only fields present in the job record are shown. The separate company file has not been joined, so other details are unknown.</p>
      <dl className="kv kv--wide">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd className={v ? "" : "muted"}>{v ?? "Unknown"}</dd>
          </div>
        ))}
      </dl>
    </section>
    </>
  );
}
