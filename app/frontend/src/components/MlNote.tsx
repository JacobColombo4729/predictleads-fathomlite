import { useAdmin } from "../lib/admin";
import { noteFor, type NoteId } from "../lib/mlNotes";

interface Props {
  id: NoteId;
  /** Picks the wording for a particular setup (the ranker the server runs, or how the resume is read). */
  variant?: string;
}

const ROWS: { key: "what" | "why" | "data" | "check" | "limits"; label: string }[] = [
  { key: "what", label: "The approach" },
  { key: "why", label: "Why we chose it" },
  { key: "data", label: "The data it uses" },
  { key: "check", label: "How it was checked" },
  { key: "limits", label: "Where it is weak" },
];

/** A collapsed "How this works" box for a machine-learning part of the app. It shows only in admin mode (Settings). */
export default function MlNote({ id, variant }: Props) {
  const { admin } = useAdmin();
  if (!admin) return null;
  const note = noteFor(id, variant);
  return (
    <details className="mlnote">
      <summary>
        <span className="mlnote__tag">Admin</span>
        <span className="mlnote__title">How this works: {note.title}</span>
      </summary>
      <dl className="mlnote__body">
        {ROWS.map(({ key, label }) => (
          <div key={key} className="mlnote__row">
            <dt>{label}</dt>
            <dd>{note[key]}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}
