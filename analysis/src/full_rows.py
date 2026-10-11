"""Full-scale EDA_SDA_1: the cleaned rows of every English US listing (src/stream_full.py --rows, about 13 million)
within 8 GB of RAM.

The analysis holds only light columns in memory (dates, numbers, flags, and repeated text as categories). Text and
list columns stay on disk and are added back, part by part, when the final table is saved.

    load_rows(columns)    the given columns, in table order (row position = the DataFrame index)
    row_extras()          per listing, built part by part: id and repost fingerprints, unified contract types,
                          tags and categories as counts and bit flags; plus the label lists the bits refer to
    save_final(df, path)  the final table as Parquet: df's columns plus the text and list columns taken from the rows
                          table by row position, written part by part
    with_missing, cat_contains, cat_map, relabel   label work done on a categorical column's categories, not on 13M strings
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.pl_data import FLAT_DIR

ROWS_TABLE = "all_fields_rows_full"
LIST_COLS = ["contract_types", "tags", "categories"]
# Text and list columns kept out of memory and added back by save_final (title_norm is rebuilt from title)
DISK_COLS = ["id", "title", "job_title", "job_title_fallback", "desc_hash", "contract_types", "tags", "categories",
             "has_recruiter", "last_processed", "salary_currency", "salary_low", "salary_high", "was_translated"]


def parts(root=None, table=ROWS_TABLE):
    found = sorted((Path(root or FLAT_DIR) / table).glob("part_*.parquet"))
    if not found:
        raise FileNotFoundError(f"No parts in {table}/: run `python -m src.stream_full --rows` first.")
    return found


def load_rows(columns, root=None, table=ROWS_TABLE):
    """Only the given columns, part by part in file order; repeated text as categories, flags as nullable booleans."""
    if set(columns) & set(LIST_COLS):
        raise ValueError("list columns are summarized by row_extras(), not loaded")
    t = pa.concat_tables(pq.read_table(p, columns=columns) for p in parts(root, table))
    out = t.to_pandas(strings_to_categorical=True, types_mapper={pa.bool_(): pd.BooleanDtype()}.get,
                      self_destruct=True, split_blocks=True)
    del t
    return out


def fingerprint(text):
    """64-bit fingerprint of each string (nullable Int64; missing stays missing)."""
    known = text.notna().to_numpy()
    fp = pd.array(pd.util.hash_array(text.fillna("").to_numpy(object)).view("int64"), dtype="Int64")
    fp[~known] = pd.NA
    return fp


def unify_label(x):
    """contract_types mixes spellings of the same label ("full_time" and "full time")."""
    return str(x).strip().lower().replace("_", " ").replace("-", " ")


def _bits(lists, labels):
    """Bit flags of each list's labels (int64; up to 63 labels, the rest counted as other)."""
    index = {label: i for i, label in enumerate(labels[:63])}
    return np.fromiter((sum(1 << index[x] for x in set(v) if x in index) for v in lists), dtype="int64", count=len(lists))


def row_extras(root=None, table=ROWS_TABLE):
    """Per listing, in table order: id_fp, has_title, repost_fp (company, normalized title and description fingerprint; missing when
    any is missing, as in 6.4), contract_bits, n_tags, category_bits, n_categories. Two passes over the parts: the first
    finds every label, the second builds the columns. Also returns the label lists and raw-label counts."""
    files = parts(root, table)
    raw_contract, contract_labels, category_labels, tag_labels = Counter(), set(), set(), set()
    for p in files:  # pass 1: labels only
        d = pq.read_table(p, columns=LIST_COLS).to_pydict()
        for v in d["contract_types"]:
            raw_contract.update(v or [])
        for v in d["categories"]:
            category_labels.update(v or [])
        for v in d["tags"]:
            tag_labels.update(v or [])
    contract_labels = sorted({unify_label(x) for x in raw_contract})
    category_labels = sorted(category_labels)
    frames = []
    for p in files:  # pass 2: one row per listing
        d = pq.read_table(p, columns=["id", "company_id", "title", "desc_hash"] + LIST_COLS).to_pandas()
        title_norm = d.title.astype("string").str.lower().str.split().str.join(" ")
        key = d.company_id.astype("string") + "\x1f" + title_norm + "\x1f" + d.desc_hash.astype("string")
        contracts = [[unify_label(x) for x in (v if v is not None else [])] for v in d.contract_types]
        cats = [list(v) if v is not None else [] for v in d.categories]
        frames.append(pd.DataFrame({
            "id_fp": fingerprint(d.id.astype("string")),
            "has_title": title_norm.fillna("").str.len().gt(0).to_numpy(),
            "repost_fp": fingerprint(key),
            "contract_bits": _bits(contracts, contract_labels),
            "n_tags": d.tags.map(lambda v: 0 if v is None else len(v)).astype("int16"),
            "category_bits": _bits(cats, category_labels),
            "n_categories": np.array([len(v) for v in cats], dtype="int16"),
        }))
    extras = pd.concat(frames, ignore_index=True)
    labels = {"contract_types": contract_labels, "categories": category_labels,
              "raw_contract_labels": len(raw_contract), "distinct_tags": len(tag_labels)}
    return extras, labels


def bit_counts(bits, labels):
    """How many listings carry each label (a listing can have several)."""
    bits = np.asarray(bits, dtype="int64")
    return pd.Series({label: int(((bits >> i) & 1).sum()) for i, label in enumerate(labels[:63])}).sort_values(ascending=False)


def has_label(bits, labels, label):
    return pd.Series((np.asarray(bits, dtype="int64") >> labels.index(label)) & 1, index=getattr(bits, "index", None)).astype(bool)


# ---------- label work on categories ----------

def with_missing(s, label="(missing)"):
    """Missing values shown as a label (categoricals need the label added as a category first)."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        if label not in s.cat.categories:
            s = s.cat.add_categories([label])
        return s.fillna(label)
    return s.fillna(label)


def cat_contains(s, pattern):
    """str.contains on a categorical column, done once per category (False where missing)."""
    hit = pd.Series(s.cat.categories.astype(str).str.contains(pattern, regex=True), dtype=bool).to_numpy()
    codes = s.cat.codes.to_numpy()
    return pd.Series(np.where(codes >= 0, hit[np.maximum(codes, 0)], False), index=s.index)


def cat_map(s, mapping):
    """A value per row from its category: mapping is a function or a dict/Series keyed by category (NaN where missing)."""
    cats = s.cat.categories
    values = cats.map(mapping) if callable(mapping) else pd.Series(mapping).reindex(cats)
    values = np.asarray(values, dtype="float64")
    codes = s.cat.codes.to_numpy()
    return pd.Series(np.where(codes >= 0, values[np.maximum(codes, 0)], np.nan), index=s.index)


def relabel(s, func):
    """Apply func to each category (e.g. trim and lowercase, or O*NET code -> role bucket); categories that become equal
    are merged, and a category func maps to None becomes missing."""
    new = [func(c) for c in s.cat.categories]
    uniq = pd.Index(pd.unique(pd.Series([v for v in new if v is not None and not pd.isna(v)], dtype=object)))
    lookup = np.array([uniq.get_loc(v) if v is not None and not pd.isna(v) else -1 for v in new] + [-1], dtype="int64")
    codes = s.cat.codes.to_numpy()
    mapped = lookup[np.where(codes >= 0, codes, len(new))]
    return pd.Series(pd.Categorical.from_codes(mapped, categories=uniq), index=s.index, name=s.name)


# ---------- scope-only fields (from the stream's scope parts, which keep every original field) ----------

SCOPE_TABLE = "job_openings_full"
EXTRA_TABLE = "job_openings_extra"  # the TripleTen roles outside computer and mathematical (--extra)
SCOPE_FIELDS = {  # flattened column -> short name; recruiter names and contacts are never kept
    "attributes.recruiter_data.title": "recruiter_title",
    "attributes.salary": "salary_text",
    "attributes.location": "location_text",
    "attributes.url": "url",
    "type": "record_type",
    "relationships.company.data.type": "company_record_type",
}


def ids_at(positions, root=None, table=ROWS_TABLE):
    """Listing ids for the given row positions of the rows table (a Series indexed by position)."""
    pos = np.sort(np.asarray(positions))
    out, start = [], 0
    for p in parts(root, table):
        n = pq.read_metadata(p).num_rows
        lo, hi = np.searchsorted(pos, [start, start + n])
        if hi > lo:
            ids = pq.read_table(p, columns=["id"]).column(0).take(pa.array(pos[lo:hi] - start)).to_pylist()
            out.append(pd.Series(ids, index=pos[lo:hi], dtype="string"))
        start += n
    return pd.concat(out) if out else pd.Series(dtype="string")


def _text(v):
    """A text field as a string; lists and other values written out, missing stays missing."""
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return str(list(v) if isinstance(v, np.ndarray) else v)


def scope_fields(ids, desc_path, root=None):
    """Fields only the scope listings carry (ids: a Series of listing ids indexed by row position), read part by part from
    the stream's scope parts:
      recruiter_key   an anonymous key per recruiter (fingerprint of name and company with a random salt that is made
                      fresh each run and never saved, so names can't be recovered); missing when no recruiter is named
      SCOPE_FIELDS    recruiter title, pay and location as written, the url, and the two record types
    The full description goes to desc_path (Parquet: id, description), not into memory. Returns a DataFrame indexed by
    row position (missing where a listing isn't in the scope parts)."""
    import os
    want = set(ids.dropna())
    pos_by_id = pd.Series(ids.index, index=ids.to_numpy())
    salt = os.urandom(16).hex()
    files = [p for t in (SCOPE_TABLE, EXTRA_TABLE) for p in sorted((Path(root or FLAT_DIR) / t).glob("part_*.pkl"))]
    if not files:
        raise FileNotFoundError(f"No parts in {SCOPE_TABLE}/: run `python -m src.stream_full` first.")
    writer = pq.ParquetWriter(desc_path, pa.schema([("id", pa.string()), ("description", pa.string())]))
    frames, seen = [], set()
    try:
        for p in files:
            d = pd.read_pickle(p)
            keep = [i in want and i not in seen for i in d["id"]]  # set lookups: fast for a million ids
            d = d[keep].drop_duplicates("id")
            if d.empty:
                continue
            seen.update(d["id"])
            col = lambda c: d.reindex(columns=[c])[c]
            name = col("attributes.recruiter_data.name").map(_text).astype("string").str.strip().str.lower()
            key = salt + "\x1f" + name + "\x1f" + col("relationships.company.data.id").astype("string")
            out = pd.DataFrame({"id": d["id"].to_numpy(), "recruiter_key": fingerprint(key.reset_index(drop=True))})
            for c, short in SCOPE_FIELDS.items():
                out[short] = col(c).map(_text).to_numpy()
            frames.append(out)
            writer.write_table(pa.table({"id": d["id"].astype(str).to_numpy(),
                                         "description": col("attributes.description").map(_text).to_numpy()},
                                        schema=writer.schema))
            del d
    finally:
        writer.close()
    out = pd.concat(frames, ignore_index=True)
    out.index = pos_by_id.loc[out.pop("id")].to_numpy()
    for short in SCOPE_FIELDS.values():
        out[short] = out[short].astype("string")
    return out.reindex(ids.index)


def scope_values(ids, column, fn=None, root=None):
    """One original field for the given listing ids (a set or Series), read part by part from the stream's scope parts
    (job_openings_full/ and job_openings_extra/): a Series indexed by id, fn applied to each value if given."""
    want = set(pd.Series(list(ids)).dropna())
    out = []
    for p in [p for t in (SCOPE_TABLE, EXTRA_TABLE) for p in sorted((Path(root or FLAT_DIR) / t).glob("part_*.pkl"))]:
        d = pd.read_pickle(p).reindex(columns=["id", column])
        d = d[[i in want for i in d["id"]]]
        if len(d):
            out.append(d.drop_duplicates("id").set_index("id")[column].map(fn) if fn else d.drop_duplicates("id").set_index("id")[column])
    s = pd.concat(out) if out else pd.Series(dtype=object)
    return s[~s.index.duplicated()]


# ---------- saving ----------

def save_final(df, path, root=None, table=ROWS_TABLE, row_group=500_000):
    """Write df (categorical columns as plain text: a dictionary per row group would repeat every category) (indexed by row position in the rows table) as Parquet, adding DISK_COLS and title_norm from the rows
    table, part by part in file order. Contract types are saved with unified labels."""
    df = df.sort_index()
    cats = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    pos = df.index.to_numpy()
    files = parts(root, table)
    disk = pq.read_schema(files[0])
    schema = pa.schema([disk.field("id"), pa.field("row", pa.int64()),
                        *pa.Schema.from_pandas(df.iloc[:0].astype({c: "string" for c in cats}), preserve_index=False),
                        *[disk.field(c) for c in DISK_COLS if c != "id"], pa.field("title_norm", pa.string())])
    writer, start = pq.ParquetWriter(path, schema), 0
    try:
        for p in files:
            extra = pq.read_table(p, columns=DISK_COLS)
            end = start + extra.num_rows
            lo, hi = np.searchsorted(pos, [start, end])
            if hi > lo:
                chunk = df.iloc[lo:hi].astype({c: "string" for c in cats})  # categories as text (see above)
                take = pa.array(pos[lo:hi] - start)
                extra = extra.take(take).to_pandas()
                extra["title_norm"] = extra.title.astype("string").str.lower().str.split().str.join(" ")
                extra["contract_types"] = extra.contract_types.map(
                    lambda v: sorted({unify_label(x) for x in v}) if v is not None else [])
                out = pd.concat([extra[["id"]], chunk.reset_index(drop=True),
                                 extra.drop(columns="id")], axis=1)
                out.insert(1, "row", pos[lo:hi])
                writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False), row_group_size=row_group)
            start = end
    finally:
        writer.close()
    return path
