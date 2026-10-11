import { useSignalPattern } from "../../lib/signals";
import PatternBadge from "./PatternBadge";

/** "Company hiring: burst / sustained" for a company id, or nothing when it has no notable pattern. */
export default function CompanyHiringBadge({ companyId }: { companyId: string | null | undefined }) {
  const pattern = useSignalPattern(companyId);
  if (!pattern) return null;
  return <span className="sg-inline"><PatternBadge pattern={pattern} compact /></span>;
}
