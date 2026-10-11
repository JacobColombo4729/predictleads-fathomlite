import { useState } from "react";
import type { FeedbackReason } from "../types";
import { NOTE_MAX, REASONS } from "../lib/feedback";

interface Props {
  title: string;
  company: string;
  /** Sends the reason (and note) to the server. Rejects if it could not be saved. */
  onSend: (reason: FeedbackReason, note: string) => Promise<void>;
  onUndo: () => void;
  onClose: () => void;
}

/** Shown after a thumbs-down: one tap says why, an optional note adds detail, and skipping costs nothing. */
export default function ReasonPrompt({ title, company, onSend, onUndo, onClose }: Props) {
  const [reason, setReason] = useState<FeedbackReason | null>(null);
  const [note, setNote] = useState("");
  const [state, setState] = useState<"idle" | "saving" | "saved" | "failed">("idle");

  const send = async (r: FeedbackReason, n: string) => {
    setState("saving");
    try {
      await onSend(r, n.trim());
      setState("saved");
    } catch {
      setState("failed");
    }
  };

  const choose = (r: FeedbackReason) => {
    setReason(r);
    void send(r, note);
  };

  return (
    <div className="reason" role="dialog" aria-label="Why was this not for you?">
      <div className="reason__head">
        <p className="reason__title">
          Marked “{title}” ({company}) as not for you. <strong>Why?</strong>
        </p>
        <button type="button" className="reason__x" aria-label="Skip" onClick={onClose}>×</button>
      </div>

      <div className="reason__chips" role="group" aria-label="Reason">
        {REASONS.map((r) => (
          <button key={r.id} type="button" className={`reason__chip${reason === r.id ? " is-on" : ""}`} aria-pressed={reason === r.id} onClick={() => choose(r.id)}>
            {r.label}
          </button>
        ))}
      </div>

      <label className="reason__note">
        <span>Anything else? (optional)</span>
        <textarea
          rows={2}
          maxLength={NOTE_MAX}
          value={note}
          placeholder="For example: I want something more hands-on."
          onChange={(e) => { setNote(e.target.value); if (state === "saved") setState("idle"); }}
        />
      </label>
      <p className="reason__hint">What you type here is used for our research on how to improve the matching.</p>

      <div className="reason__row">
        <button type="button" className="btn btn--sm" disabled={!note.trim() || state === "saving"} onClick={() => void send(reason ?? "other", note)}>
          Save note
        </button>
        <button type="button" className="btn btn--sm" onClick={onUndo}>Undo 👎</button>
        <button type="button" className="btn btn--sm btn--primary" onClick={onClose}>Done</button>
        <span className="reason__status" role="status">
          {state === "saving" ? "Saving…" : state === "saved" ? "Thanks, noted." : state === "failed" ? "Could not save that. Try again." : ""}
        </span>
      </div>
    </div>
  );
}
