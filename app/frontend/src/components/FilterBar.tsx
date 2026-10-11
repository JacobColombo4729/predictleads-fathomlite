import { useEffect, useState, type FormEvent } from "react";
import type { DescriptionFilter, Filters } from "../types";
import { SearchIcon } from "./Icons";

interface Props {
  filters: Filters;
  onChange: (next: Filters) => void;
}

export default function FilterBar({ filters, onChange }: Props) {
  // The search words are applied with the Search button (or Enter); location and description apply as you change them.
  const [draft, setDraft] = useState(filters.query);
  useEffect(() => setDraft(filters.query), [filters.query]);

  const set = <K extends keyof Filters>(key: K, value: Filters[K]) =>
    onChange({ ...filters, [key]: value });

  const submit = (e: FormEvent) => {
    e.preventDefault();
    set("query", draft.trim());
  };
  const clear = () => {
    setDraft("");
    set("query", "");
  };

  return (
    <div className="filters">
      <form className="searchform" role="search" onSubmit={submit}>
        <label className="search">
          <SearchIcon />
          <span className="sr-only">Search jobs by title or keywords</span>
          <input
            type="search"
            value={draft}
            placeholder="Search jobs by title or keywords, e.g. sales engineer"
            onChange={(e) => setDraft(e.target.value)}
          />
        </label>
        <button type="submit" className="btn btn--primary">Search</button>
        {filters.query && <button type="button" className="btn btn--ghost" onClick={clear}>Clear</button>}
      </form>
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
