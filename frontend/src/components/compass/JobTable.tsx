import CompanyHiringBadge from "../signals/CompanyHiringBadge";
import type { CompassJob, CompassResult } from "../../types";
import { TIER_NAME, tierOf } from "../../lib/compass";

interface Props {
  result: CompassResult;
  themeFilter: string;
  stretch: number;
  onSelect: (jobId: string) => void;
}

/** The map as a table, so nothing depends on seeing positions or colours. */
export default function JobTable({ result, themeFilter, stretch, onSelect }: Props) {
  const rows = result.jobs.filter((j: CompassJob) => themeFilter === "all" || j.themeId === themeFilter).sort((a, b) => b.fit - a.fit);
  return (
    <table className="cp-table">
      <thead>
        <tr><th scope="col">Job</th><th scope="col">Company</th><th scope="col">Theme</th><th scope="col">Tier</th><th scope="col">Fit</th><th scope="col">Type</th></tr>
      </thead>
      <tbody>
        {rows.map((j) => (
          <tr key={j.job.id} onClick={() => onSelect(j.job.id)}>
            <td>{j.job.title}</td>
            <td>{j.job.company.name}<CompanyHiringBadge companyId={j.job.company.id} /></td>
            <td>{result.themes.find((t) => t.id === j.themeId)?.name}</td>
            <td>{TIER_NAME[tierOf(j.fit, stretch)]}</td>
            <td>{j.fit}%</td>
            <td>{j.pick ? "Picked" : "Suggested"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
