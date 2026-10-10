import type { DescriptionFilter, Filters } from "../types";
import { SearchIcon } from "./Icons";

interface Props {
  filters: Filters;
  onChange: (next: Filters) => void;
}

export default function FilterBar({ filters, onChange }: Props) {
  const set = <K extends keyof Filters>(key: K, value: Filters[K]) =>
    onChange({ ...filters, [key]: value });

  return (
    <div className="filters">
      <label className="search">
        <SearchIcon />
        <span className="sr-only">Narrow results by words in the job title or description</span>
        <input
          type="search"
          value={filters.query}
          placeholder="Narrow by words in the title or description"
          onChange={(e) => set("query", e.target.value)}
        />
      </label>
      <div className="filters__row">
        <label className="select">
          <span>Location contains</span>
          <input
            type="text"
            value={filters.location}
            placeholder="e.g. Texas or Remote"
            onChange={(e) => set("location", e.target.value)}
          />
        </label>
        <label className="select">
          <span>Description</span>
          <select
            value={filters.description}
            onChange={(e) => set("description", e.target.value as DescriptionFilter)}
          >
            <option value="any">Any</option>
            <option value="with">Has description</option>
            <option value="without">No description</option>
          </select>
        </label>
      </div>
      <p className="muted small">Work arrangement is not in this dataset, so it can't be filtered.</p>
    </div>
  );
}
