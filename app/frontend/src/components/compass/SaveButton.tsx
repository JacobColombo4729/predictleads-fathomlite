import type { CompassJob } from "../../types";
import { BookmarkIcon } from "../Icons";

interface Props {
  job: CompassJob;
  saved: boolean;
  onSave: (job: CompassJob) => void;
}

/** A small bookmark that sends a role to the Saved list. It fills in once saved; remove it from the Saved tab. */
export default function SaveButton({ job, saved, onSave }: Props) {
  return (
    <button
      type="button"
      className={`cp-save${saved ? " is-saved" : ""}`}
      aria-pressed={saved}
      aria-label={saved ? `${job.job.title} is saved` : `Save ${job.job.title} to your Saved jobs`}
      title={saved ? "Saved. Remove it from the Saved tab." : "Save to your Saved jobs"}
      disabled={saved}
      onClick={(e) => { e.stopPropagation(); onSave(job); }}
    >
      <BookmarkIcon filled={saved} width={16} height={16} />
    </button>
  );
}
