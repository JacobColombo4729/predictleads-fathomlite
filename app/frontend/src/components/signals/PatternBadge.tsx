import type { SignalPattern } from "../../types";
import { PATTERN_HELP, PATTERN_ICON, PATTERN_LABEL } from "../../lib/signals";

/** Pattern name with an icon, so the meaning never rests on colour alone. */
export default function PatternBadge({ pattern, compact = false }: { pattern: SignalPattern; compact?: boolean }) {
  return (
    <span className={`sg-badge sg-badge--${pattern}`} title={PATTERN_HELP[pattern]}>
      <span aria-hidden="true">{PATTERN_ICON[pattern]}</span>
      {compact ? `Company hiring: ${PATTERN_LABEL[pattern].toLowerCase()}` : PATTERN_LABEL[pattern]}
    </span>
  );
}
