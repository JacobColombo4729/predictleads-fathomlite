"""O*NET tables and skills extraction (EDA_SDA_3 section 6, then ML_Fit's skills-based matching).

Data: the O*NET 31.0 Database, U.S. Department of Labor, Employment and Training Administration (USDOL/ETA), used under
CC BY 4.0. The tab-separated tables come from `python -m fetch.get_onet` into `external("onet")`.

Three parts:
  1. Readers for the tables (only the needed columns; suppressed and not-relevant ratings dropped).
  2. A skill vocabulary (software tools plus skill element names) and a phrase matcher on normalized text. Matching is on
     whole words, so "java" is not found inside "javascript". Aliases ("js" for javascript) apply to the vocabulary and to
     the text alike.
  3. `extract`: reads each listing's cleaned text (eda3_desc_clean.parquet) batch by batch, looks only at its requirement
     lines, and writes (id, name, kind, section, element_id) rows, marked required or preferred, with the same crash-safe
     .part writes as text_features.build.

Nothing here prints listing text.
"""
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if str(Path(__file__).resolve().parents[2]) not in sys.path:  # the repository root, so fetch (shared with app/) imports
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fetch.paths import external  # noqa: E402
from src import text_features as tf

CREDIT = "O*NET 31.0 Database, USDOL/ETA (CC BY 4.0)"

# ---------- 1. readers ----------
RATING_COLS = ["O*NET-SOC Code", "Element ID", "Element Name", "Scale ID", "Data Value", "Recommend Suppress", "Not Relevant"]
SCALES = ("IM", "LV")  # importance and level


def read_table(name, usecols=None, folder=None):
    """One O*NET table ("<name>.txt": UTF-8, tab-separated, header on the first line) with every column as text.
    Quotes are not special in these files, so they are read as plain characters."""
    folder = Path(folder) if folder is not None else external("onet")
    return pd.read_csv(folder / f"{name}.txt", sep="\t", dtype=str, usecols=usecols, encoding="utf-8", quoting=3,
                       keep_default_na=False)


def occupations(folder=None):
    """O*NET-SOC Code, Title, Description."""
    return read_table("Occupation Data", ["O*NET-SOC Code", "Title", "Description"], folder)


def ratings(name, folder=None):
    """Essential Skills or Transferable Skills: importance (IM) and level (LV) rows, with suppressed and not-relevant
    ratings dropped. Columns: code, element_id, element, scale, value (float)."""
    d = read_table(name, RATING_COLS, folder)
    d = d[d["Scale ID"].isin(SCALES) & (d["Recommend Suppress"] != "Y") & (d["Not Relevant"] != "Y")]
    d = d.assign(value=pd.to_numeric(d["Data Value"], errors="coerce")).dropna(subset=["value"])
    return d.rename(columns={"O*NET-SOC Code": "code", "Element ID": "element_id", "Element Name": "element",
                             "Scale ID": "scale"})[["code", "element_id", "element", "scale", "value"]].reset_index(drop=True)


def software(folder=None):
    """Software Skills: code, tool (Workplace Example), commodity (Element Name), element_id, hot, in_demand (booleans)."""
    d = read_table("Software Skills", ["O*NET-SOC Code", "Workplace Example", "Element ID", "Element Name", "Hot Technology", "In Demand"], folder)
    return pd.DataFrame({"code": d["O*NET-SOC Code"], "tool": d["Workplace Example"], "commodity": d["Element Name"],
                         "element_id": d["Element ID"], "hot": d["Hot Technology"].eq("Y"), "in_demand": d["In Demand"].eq("Y")})


def job_zones(folder=None):
    """Job Zones: code, job_zone (int)."""
    d = read_table("Job Zones", ["O*NET-SOC Code", "Job Zone"], folder)
    return pd.DataFrame({"code": d["O*NET-SOC Code"], "job_zone": pd.to_numeric(d["Job Zone"], errors="coerce")}).dropna()


def related(folder=None):
    """Related Occupations: code, related, tier, index."""
    d = read_table("Related Occupations", ["O*NET-SOC Code", "Related O*NET-SOC Code", "Relatedness Tier", "Index"], folder)
    return d.rename(columns={"O*NET-SOC Code": "code", "Related O*NET-SOC Code": "related", "Relatedness Tier": "tier", "Index": "index"})


def content_model(folder=None):
    """Content Model Reference: element_id, element, description."""
    d = read_table("Content Model Reference", ["Element ID", "Element Name", "Description"], folder)
    return d.rename(columns={"Element ID": "element_id", "Element Name": "element", "Description": "description"})


# ---------- 2. normalizing and the vocabulary ----------
ALIASES = {  # spelled-out forms to one canonical form, on normalized text (applied to the vocabulary and the listings)
    "js": "javascript", "java script": "javascript", "node js": "nodejs", "vue js": "vuejs",
    "react js": "reactjs", "k8s": "kubernetes", "postgres": "postgresql", "postgre sql": "postgresql", "mongo": "mongodb",
    "gcp": "google cloud platform", "powerbi": "power bi", "sklearn": "scikit learn",
    "ms excel": "microsoft excel", "ms word": "microsoft word", "ms office": "microsoft office",
    "ms powerpoint": "microsoft powerpoint", "ms access": "microsoft access", "ms sql server": "microsoft sql server",
    "mssql": "microsoft sql server", "amazon web services": "aws",
}
# Tool names that are everyday words or single letters: matching them as phrases would fire on ordinary sentences
# ("go to market", "word of mouth"). Dropped from the vocabulary; the notebook reports how many names were skipped.
AMBIGUOUS = {"spark", "chef", "puppet", "word", "access", "outlook", "project", "teams", "office",
             "sketch", "unity", "express", "view", "flow", "pivot", "base", "tableau public"}
# Language names that are also everyday words or letters (R, Go, C, Swift): kept, but matched only in context, i.e.
# next to another programming or analysis tool or the word "programming" / "language" ("Python or R", "C/C++",
# "the Go programming language"), or through an unambiguous spelling (golang, rstudio, swiftui).
CONTEXTUAL = {"r", "go", "c", "swift"}
SPELLINGS = {"golang": "go", "rstudio": "r", "swiftui": "swift"}
NEIGHBORS = {"python", "java", "javascript", "typescript", "sql", "sas", "stata", "spss", "matlab", "scala", "julia",
             "cpp", "csharp", "rust", "kotlin", "ruby", "php", "perl", "bash", "fortran", "haskell", "nodejs", "excel",
             "tableau", "programming", "language", "languages", "lang", *CONTEXTUAL, *SPELLINGS}
CONTEXT_WINDOW = 3  # tokens on each side
MIN_PHRASE_CHARS = 2
_ALIAS_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(a) for a in sorted(ALIASES, key=len, reverse=True)) + r")(?![a-z0-9])")


def normalize(text):
    """Lowercase words separated by single spaces. C++, C#, F# and .NET keep their meaning ("cpp", "csharp", "fsharp",
    "dotnet"); dots inside a word are dropped ("Node.js" -> "nodejs"); other punctuation becomes a space; aliases
    are replaced by their canonical form."""
    x = str(text).lower().replace("’", "").replace("'", "")
    x = x.replace("c++", "cpp ").replace("c#", "csharp ").replace("f#", "fsharp ").replace("&", " and ")
    x = re.sub(r"(?<![a-z0-9])\.net\b", "dotnet", x)
    x = re.sub(r"(?<=[a-z0-9])\.(?=[a-z0-9])", "", x)
    x = " ".join(re.sub(r"[^a-z0-9]+", " ", x).split())
    return _ALIAS_RE.sub(lambda m: ALIASES[m.group(1)], x)


def vocabulary(software_table, skill_tables, extra_exclude=()):
    """The skill vocabulary: one row per phrase. Columns: phrase (normalized), name (as O*NET writes it), kind
    ("tool" for a Workplace Example, "skill" for a skill element name), element_id (the commodity or skill element),
    hot, in_demand (tools), n_occupations (how many O*NET occupations list it). Phrases that are very short or
    ambiguous words (AMBIGUOUS, extra_exclude) are left out and counted in `.attrs["skipped"]`."""
    t = software_table
    tools = (t.groupby("tool", sort=True).agg(commodity=("commodity", "first"), element_id=("element_id", "first"),
                                              hot=("hot", "max"), in_demand=("in_demand", "max"), n_occupations=("code", "nunique"))
             .reset_index().rename(columns={"tool": "name"}))
    tools["kind"] = "tool"
    parts = [tools.drop(columns="commodity")]
    skills = pd.concat(skill_tables, ignore_index=True)
    sk = skills.groupby("element", sort=True).agg(element_id=("element_id", "first"), n_occupations=("code", "nunique")).reset_index()
    sk = sk.rename(columns={"element": "name"}).assign(kind="skill", hot=False, in_demand=False)
    parts.append(sk)
    v = pd.concat(parts, ignore_index=True)
    v["phrase"] = v["name"].map(normalize)
    v["context"] = v.phrase.isin(CONTEXTUAL) & v.kind.eq("tool")  # matched only in context (see CONTEXTUAL)
    excluded = AMBIGUOUS | {normalize(x) for x in extra_exclude}
    drop = ~v.context & ((v.phrase.str.len() < MIN_PHRASE_CHARS) | v.phrase.isin(excluded | CONTEXTUAL))
    out = v[~drop].drop_duplicates(["phrase", "name", "kind"]).reset_index(drop=True)
    out.attrs["skipped"] = int(drop.sum())
    return out[["phrase", "name", "kind", "element_id", "hot", "in_demand", "n_occupations", "context"]]


class Matcher:
    """Finds vocabulary phrases in normalized text as whole words. Built once, then used on every line; only tokens that
    start some phrase are looked at, so the cost per line stays small."""

    def __init__(self, vocab):
        self.entries = {}     # phrase tokens -> [(name, kind, element_id)]
        self.reach = {}       # first token -> longest phrase (in tokens) starting with it
        self.contextual = {}  # R, Go, C, Swift: token -> [(name, kind, element_id)], matched only in context
        ctx = vocab["context"] if "context" in vocab else pd.Series(False, index=vocab.index)
        for phrase, name, kind, eid, c in zip(vocab.phrase, vocab.name, vocab.kind, vocab.element_id, ctx):
            toks = tuple(phrase.split())
            if not toks:
                continue
            if c:
                self.contextual.setdefault(phrase, []).append((name, kind, eid))
                continue
            self.entries.setdefault(toks, []).append((name, kind, eid))
            self.reach[toks[0]] = max(self.reach.get(toks[0], 0), len(toks))
        self.fingerprint = hashlib.blake2b(
            "\n".join(sorted([" ".join(k) + "|" + "|".join(f"{n}/{kd}" for n, kd, _ in v) for k, v in self.entries.items()]
                             + [f"~{k}|" + "|".join(f"{n}/{kd}" for n, kd, _ in v) for k, v in self.contextual.items()])).encode(),
            digest_size=8).hexdigest()

    def find(self, text):
        """The set of (name, kind, element_id) whose phrase appears in `text` (any case, any punctuation)."""
        toks = normalize(text).split()
        found = set()
        for i, t in enumerate(toks):
            if t in SPELLINGS:
                found.update(self.contextual.get(SPELLINGS[t], ()))
            elif t in self.contextual:
                window = toks[max(0, i - CONTEXT_WINDOW):i] + toks[i + 1:i + 1 + CONTEXT_WINDOW]
                if any(w in NEIGHBORS for w in window):
                    found.update(self.contextual[t])
            reach = self.reach.get(t)
            if reach:
                for n in range(1, min(reach, len(toks) - i) + 1):
                    hit = self.entries.get(tuple(toks[i:i + n]))
                    if hit:
                        found.update(hit)
        return found


# ---------- 3. requirement lines and the batch extraction ----------
OTHER_HEAD = re.compile(r"(?:responsibilit\w*|duties|what you(?:'ll| will) (?:do|be doing)|the role|the job|about|benefits|perks|compensation"
                        r"|overview|description|our team|why join)", re.IGNORECASE)


def requirement_lines(lines):
    """(line, section) for the requirement lines of one listing, section "required" or "preferred" (tf.line_sections).
    A requirement line sits under a requirements-style or preferred-style heading (tf.REQ_HEAD, tf.PREF_HEAD) until a
    responsibilities-style heading (OTHER_HEAD) or the next requirements heading, or carries a requirement cue
    (tf.REQUIREMENT_CUE: years, experience, proficiency, skills, "must", ...). Responsibilities and other lines are not
    read. A short line such as "Python" is a list item, not a heading, unless it names a section; unheaded text with a
    cue counts as required, as in tf.line_sections."""
    sections = tf.line_sections(lines)
    out, in_req = [], False
    for x, s in zip(lines, sections):
        if len(x.split()) <= 6 and tf.HEADING.match(x):
            if tf.REQ_HEAD.search(x) or tf.PREF_HEAD.search(x):
                in_req = True
            elif OTHER_HEAD.search(x):
                in_req = False
        if in_req or tf.REQUIREMENT_CUE.search(x):
            out.append((x, s))
    return out


def listing_skills(text, matcher):
    """{(name, kind, element_id): "required" | "preferred"} for one cleaned description. A skill asked on any
    required line is required; one found only on preferred lines is preferred."""
    found = {}
    for line, section in requirement_lines(text.split("\n") if text else []):
        for key in matcher.find(line):
            if found.get(key) != "required":
                found[key] = section
    return found


SKILLS_SCHEMA = pa.schema([("id", pa.string()), ("name", pa.string()), ("kind", pa.string()), ("section", pa.string()),
                           ("element_id", pa.string())])

_MATCHER = None


def _init_worker(matcher):
    global _MATCHER
    _MATCHER = matcher


def _extract_batch(job):
    ids, texts = job
    rows = {"id": [], "name": [], "kind": [], "section": [], "element_id": []}
    for i, text in zip(ids, texts):
        for (name, kind, eid), section in listing_skills(text, _MATCHER).items():
            rows["id"].append(i)
            rows["name"].append(name)
            rows["kind"].append(kind)
            rows["section"].append(section)
            rows["element_id"].append(eid)
    return pa.table(rows, schema=SKILLS_SCHEMA)


def _metadata(clean_path, matcher):
    return {b"vocab": matcher.fingerprint.encode(), b"listings": str(pq.ParquetFile(clean_path).metadata.num_rows).encode()}


def cache_valid(skills_path, clean_path, matcher):
    """True if the skills file can be read, was built from this vocabulary and from a cleaned-text file with the same
    number of listings (a file left by a killed run, or built from other tables, is not valid)."""
    try:
        meta = pq.ParquetFile(skills_path).schema_arrow.metadata or {}
        return meta == {**meta, **_metadata(clean_path, matcher)}
    except Exception:
        return False


def extract(clean_path, matcher, skills_path, batch_size=10_000, progress=None, workers=None):
    """Find each listing's skills in its requirement lines, batch by batch (several processes, in order), and write
    them to `skills_path` as <name>.part, moved into place only when complete. Listings with no match have no rows.
    Returns the number of rows written."""
    workers = tf.default_workers() if workers is None else workers
    part = Path(str(skills_path) + ".part")
    schema = SKILLS_SCHEMA.with_metadata(_metadata(clean_path, matcher))
    jobs = ((b.column("id").to_pylist(), b.column("desc_clean").to_pylist())
            for b in pq.ParquetFile(clean_path).iter_batches(batch_size=batch_size, columns=["id", "desc_clean"]))
    writer, total = None, 0
    try:
        writer = pq.ParquetWriter(part, schema)
        for n, t in enumerate(tf._pooled(_extract_batch, jobs, workers, _init_worker, (matcher,))):
            writer.write_table(t.cast(schema))
            total += t.num_rows
            if progress:
                progress(n, t.num_rows)
        writer.close()
        writer = None
        part.replace(skills_path)
    finally:
        if writer is not None:
            writer.close()
        part.unlink(missing_ok=True)
    return total


# ---------- 4. profiles for the notebook (aggregates) ----------
def codes_like(all_codes, code):
    """O*NET-SOC codes in `all_codes` that start with `code` (a full code like "15-1252.00" or a 7-character one)."""
    c = str(code).strip()
    return [x for x in all_codes if x.startswith(c)] if c else []


def listed_by_code(listing_codes, software_table, essential):
    """For each distinct listing code: the set of tools (software) and skill elements (essential, IM rows) O*NET lists
    for it. A code matches the O*NET codes that start with it, so a 7-character code covers its detailed codes.
    Returns a long table: listing_code, name, kind."""
    tools = software_table[["code", "tool"]].rename(columns={"tool": "name"}).assign(kind="tool")
    elements = essential[essential.scale == "IM"][["code", "element"]].rename(columns={"element": "name"}).assign(kind="skill")
    listed = pd.concat([tools, elements], ignore_index=True).drop_duplicates()
    rows = []
    for lc in pd.Series(listing_codes).dropna().astype(str).str.strip().unique():
        sub = listed[listed.code.str.startswith(lc)][["name", "kind"]].drop_duplicates()
        rows.append(sub.assign(listing_code=lc))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["name", "kind", "listing_code"])


def agreement(skills, listing_code, listed):
    """Per listing: matched skills, and the share O*NET lists for the listing's own occupation.
    skills: id, name, kind rows; listing_code: Series indexed by id; listed: from listed_by_code.
    Returns a DataFrame indexed by id (only listings with a code and a match): n_matched, n_listed, share."""
    s = skills[["id", "name", "kind"]].drop_duplicates().merge(listing_code.rename("listing_code").rename_axis("id").reset_index(), on="id")
    s = s.dropna(subset=["listing_code"])
    s["listing_code"] = s.listing_code.astype(str).str.strip()
    key = listed.assign(hit=True)
    m = s.merge(key, on=["listing_code", "name", "kind"], how="left")
    g = m.groupby("id").agg(n_matched=("name", "size"), n_listed=("hit", lambda x: int(x.notna().sum())))
    g["share"] = g.n_listed / g.n_matched
    return g


def job_zone_for(listing_codes, zones):
    """Job Zone for each listing code: the exact code if O*NET has it, else the median zone of the O*NET codes that start
    with it (a 7-character code). NaN if none."""
    zmap = zones.set_index("code").job_zone
    out = {}
    for lc in pd.Series(listing_codes).dropna().astype(str).str.strip().unique():
        if lc in zmap.index:
            out[lc] = float(zmap[lc])
        else:
            hit = zmap[zmap.index.str.startswith(lc)]
            out[lc] = float(hit.median()) if len(hit) else float("nan")
    return pd.Series(listing_codes).astype(str).str.strip().map(out)


def skill_matrix(transferable, scale="IM"):
    """Occupation x skill element table of one rating scale (O*NET-SOC code rows, element name columns)."""
    d = transferable[transferable.scale == scale]
    return d.pivot_table(index="code", columns="element", values="value", aggfunc="mean")


def profile(matrix, codes, weights=None):
    """Mean rating per skill element over the O*NET occupations that start with any of `codes` (optionally weighted by
    listing counts per code, a dict). An empty Series if nothing matches."""
    parts, w = [], []
    for c in codes:
        rows = matrix[matrix.index.str.startswith(str(c).strip())]
        if len(rows):
            parts.append(rows.mean())
            w.append((weights or {}).get(c, 1.0))
    if not parts:
        return pd.Series(dtype=float)
    table, w = pd.concat(parts, axis=1, ignore_index=True), pd.Series(w, dtype=float)
    return (table * w).sum(axis=1, min_count=1) / table.notna().mul(w).sum(axis=1)  # weighted mean over occupations that rate it


def carry_over(source, target, threshold=3.0):
    """Skills both profiles rate at `threshold` importance or higher (O*NET importance runs 1-5): the skills that carry
    over from a background occupation to a target role. Returns a Series of the lower of the two ratings, highest first."""
    both = pd.concat([source, target], axis=1, keys=["s", "t"]).dropna()
    both = both[(both.s >= threshold) & (both.t >= threshold)]
    return both.min(axis=1).sort_values(ascending=False)
