import type { SignalPattern } from "../types";

export type MarketTask = "companies" | "technologies" | "listings" | "trend" | "pay" | "compare";

/** What a question asks for (the same fields the server validates). Defaults mean "no filter". */
export interface MarketQuery {
  task: MarketTask;
  role_family: ("ai_ml" | "data" | "technical_sales")[];
  states: string[];
  cities: string[];
  countries: string[];
  title_words: string[];
  technologies: string[];
  technology_match: "any" | "all";
  seniority: string[];
  patterns: ("burst" | "sustained" | "cooling")[];
  window_days: number;
  first_time_hiring: boolean;
  companies: string[];
  only_my_saved: boolean;
  sort: "new" | "active" | "surge";
  limit: number;
}

export interface MarketExample { question: string; query: Partial<MarketQuery> }

export interface MarketOverview {
  available: boolean;
  ask_enabled: boolean;
  research_enabled: boolean;
  voice_enabled: boolean;
  examples: MarketExample[];
  as_of: string | null;
  limits: { ask: number; research: number };
}

export interface Chip { label: string; value: string; fields: string[] }

export interface CompanyRow {
  company_id: string;
  name: string | null;
  domain: string | null;
  new: number;
  active: number;
  listings: number;
  pattern: SignalPattern | "quiet" | null;
  surge_z: number | null;
  company_new_30: number | null;
  example_titles: string[];
  places: string[];
  technologies: Record<string, number>;
  weekly: number[];
  top_technologies?: { name: string; listings: number }[];
  families?: Record<string, number>;
}

export interface MarketResult {
  task: MarketTask;
  rows: Record<string, unknown>[];
  summary: Record<string, number>;
  notes: string[];
  interpretation: Chip[];
  as_of: string;
  week_labels?: string[];
}

export type MarketResponse =
  | { state: "ok"; question?: string; query: MarketQuery; result: MarketResult }
  | { state: "unsupported"; question: string; reason: string }
  | { state: "unavailable"; question?: string };

export type ResearchResponse =
  | { state: "ok"; name: string; text: string; sources: { url: string; title: string }[]; label: string; retrieved: string }
  | { state: "no_sources"; message: string }
  | { state: "unavailable" };

/** A full query with defaults filled in, from a preset's partial one. */
export function fullQuery(partial: Partial<MarketQuery>): MarketQuery {
  return {
    task: "companies", role_family: [], states: [], cities: [], countries: [], title_words: [], technologies: [], technology_match: "any",
    seniority: [], patterns: [], window_days: 30, first_time_hiring: false, companies: [], only_my_saved: false, sort: "new", limit: 20,
    ...partial,
  };
}

/** The query with the fields a chip controls put back to "no filter". */
export function withoutFields(query: MarketQuery, fields: string[]): MarketQuery {
  const next: Record<string, unknown> = { ...query };
  for (const f of fields) next[f] = f === "first_time_hiring" || f === "only_my_saved" ? false : [];
  return next as unknown as MarketQuery;
}

export type TranscribeResponse = { state: "ok"; text: string } | { state: "empty"; message: string } | { state: "unavailable" };

export const money = (n: number): string => `$${Math.round(n).toLocaleString()}`;
