import type { HistoricalStatus } from "../types";

export function formatDate(iso: string): string {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function formatSavedDate(isoTimestamp: string): string {
  return new Date(isoTimestamp).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function safeHttpUrl(url: string): string | null {
  try {
    const u = new URL(url);
    return u.protocol === "https:" || u.protocol === "http:" ? u.href : null;
  } catch {
    return null;
  }
}

export const historicalStatusText: Record<HistoricalStatus, string> = {
  listed_when_observed: "Listed when observed",
  not_in_latest_snapshot: "Closed in source data",
  unknown_status: "Unknown (not stated in the source)",
};

const ENTITIES: Record<string, string> = { "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&apos;": "'" };

/** A listing's description with formatting leftovers removed, for display only (matching uses the stored text):
 *  markdown (**bold**, ### headings, \- escapes, ---- rules), separators (|, ~, ^, `), other bullet symbols (turned
 *  into the list's "- "), HTML entities, icon-font glyphs and invisible characters. "~5 years" reads "about 5 years". */
export function cleanDescription(text: string): string {
  return text
    .replace(/&(?:nbsp|amp|lt|gt|quot|apos|#39);/g, (m) => ENTITIES[m] ?? m)
    .replace(/[​-‍⁠﻿­]/g, "") // zero-width characters and soft hyphens
    .replace(/^[ \t]*[-][ \t]*/gm, "- ") // an icon-font glyph opening a line is a bullet
    .replace(/[-]/g, "") // other icon-font glyphs (private use area)
    .replace(/\\([-*_#.()[\]!+>~|])/g, "$1") // markdown escapes: \- \* \.
    .replace(/(\*\*|__)(.+?)\1/g, "$2") // **bold**, __bold__
    .replace(/^[ \t]*#{1,6}[ \t]*/gm, "") // ### headings
    .replace(/^[ \t]*[-=_*~|]{3,}[ \t]*$/gm, "") // ---- ==== |||| rules
    .replace(/^[ \t]*[•·▪◦●■►▸➢✓✔*+>][ \t]*/gm, "- ") // other bullets
    .replace(/\*(\S[^*\n]*?)\*/g, "$1") // *emphasis*
    .replace(/~(?=[ \t]?\$?\d)/g, "about ") // ~5 years, ~$80k
    .replace(/[~^`¦*]+/g, " ")
    .replace(/[ \t]*\|+[ \t]*/g, " · ") // | separators read as a middle dot
    .replace(/^[ \t]*·[ \t]*|[ \t]*·[ \t]*$/gm, "") // ...but not at the start or end of a line
    .replace(/[ \t]{2,}/g, " ")
    .replace(/^[ \t]+|[ \t]+$/gm, "");
}

export function descriptionParagraphs(text: string): { type: "p" | "ul"; items: string[] }[] {
  const blocks: { type: "p" | "ul"; items: string[] }[] = [];
  for (const raw of cleanDescription(text).split(/\n\s*\n/)) {
    const lines = raw.split("\n").map((l) => l.trim()).filter(Boolean);
    if (!lines.length) continue;
    const bullets = lines.filter((l) => l.startsWith("- "));
    if (bullets.length === lines.length) {
      blocks.push({ type: "ul", items: bullets.map((l) => l.slice(2)) });
    } else {
      blocks.push({ type: "p", items: [lines.join(" ")] });
    }
  }
  return blocks;
}
