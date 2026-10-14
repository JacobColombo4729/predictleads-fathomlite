"""Role discovery, evaluated on real career moves (ML_Fit sections 2-4 and 6).

For each career changer, rank the role buckets from their pre-switch background alone and check whether the role they
actually moved into is in the top 3 or top 5. Methods: keyword overlap, TF-IDF cosine, BGE cosine and RRF (TF-IDF and
BGE fused), against the same baseline (the most common role in the training people). Each role is described by its
2026 listings. No method is fit on people: TF-IDF learns its vocabulary from the listings and BGE is pretrained, so a
test person can't leak into a method; only the baseline learns from people, from the training split.
Prints and returns aggregates only.
"""
import hashlib
import heapq
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.career_switch import ROLE_ORDER, role_bucket
from src.pl_data import FLAT_DIR

ROLES = list(ROLE_ORDER)
SCOPE_TABLE = "job_openings_full"
TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[a-z][a-z+#.]*[a-z+#]|[a-z]")
STOP = {"and", "or", "the", "of", "in", "for", "to", "a", "an", "with", "at", "on", "&", "i", "ii", "iii", "iv", "jr",
        "sr", "senior", "junior", "lead", "level", "associate", "intern", "internship", "team", "remote", "part", "time",
        "full", "contract", "temporary"}
RRF_K = 60  # the standard reciprocal-rank-fusion constant, as in the app's ranker


# ---------- text ----------

def clean(text):
    """Lowercase plain text: HTML tags removed, whitespace collapsed."""
    return " ".join(TAG_RE.sub(" ", str(text or "")).lower().split())


def words(text):
    """Distinct content words of a text (no stopwords or seniority words)."""
    return {w for w in WORD_RE.findall(clean(text)) if w not in STOP}


# ---------- role documents: a reproducible sample of each role's listings ----------

def _key(listing_id, seed):
    """A stable pseudo-random number for a listing, so the same listings are sampled on every run and machine."""
    return int(hashlib.sha1(f"{seed}:{listing_id}".encode()).hexdigest()[:12], 16)


def role_listings(n_per_role=2000, seed=0, root=None, table=SCOPE_TABLE):
    """Up to n_per_role listings per role bucket from the full-file scope parts, chosen by a seeded hash of the id
    (reproducible, memory-light: one part at a time). Returns a DataFrame: role, title, text (title + description)."""
    cols = ["id", "attributes.title", "attributes.description", "attributes.onet_data.code"]
    keep = {r: [] for r in ROLES}  # per role, a heap of (-key, title, text) holding the n smallest keys
    parts = sorted((Path(root or FLAT_DIR) / table).glob("part_*.pkl"))
    if not parts:
        raise FileNotFoundError(f"no {table} parts under {Path(root or FLAT_DIR)}: build them first (EDA_SDA_1 cell 4b)")
    for path in parts:
        part = pd.read_pickle(path).reindex(columns=cols)
        part["role"] = part["attributes.onet_data.code"].map(role_bucket)
        for row in part[part.role.isin(ROLES)].itertuples(index=False):
            item = (-_key(row[0], seed), clean(row[1]), clean(f"{row[1]} {row[2]}"))
            heap = keep[row[-1]]
            if len(heap) < n_per_role:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
        del part
    rows = [(role, title, text) for role, heap in keep.items() for _, title, text in sorted(heap, reverse=True)]
    return pd.DataFrame(rows, columns=["role", "title", "text"])


# ---------- the methods: each returns a people x roles score matrix (higher = better fit) ----------

def keyword_scores(people_text, listings, top_words=200):
    """Keyword overlap: how many of a role's most common title words appear in the person's background."""
    vocab = {}
    for role, titles in listings.groupby("role").title:
        counts = pd.Series([w for t in titles for w in words(t)]).value_counts()
        # ties at the cutoff broken alphabetically: words() returns a set, whose order follows Python's string hash,
        # which changes with every new process, so the tied words kept (and the trained layer's choice) changed by run
        counts = counts.rename_axis("word").reset_index(name="n").sort_values(["n", "word"], ascending=[False, True])
        vocab[role] = set(counts.word.iloc[:top_words])
    person_words = [words(t) for t in people_text]
    return np.array([[len(pw & vocab.get(r, set())) for r in ROLES] for pw in person_words], dtype=float)


def tfidf_scores(people_text, listings):
    """TF-IDF cosine between the person's background and each role's average listing vector. The vocabulary and word
    weights come from the individual listings (no person is involved); a role is the mean of its listings' vectors."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.5, min_df=2, stop_words="english")
    x = vec.fit_transform(listings.text)
    centroids = np.vstack([np.asarray(x[listings.role.eq(r).to_numpy()].mean(axis=0)) if listings.role.eq(r).any()
                           else np.zeros((1, x.shape[1])) for r in ROLES])
    people = vec.transform([clean(t) for t in people_text])
    return np.asarray(people @ normalize(centroids).T)


def bge_scores(people_text, listings, n_per_role=300, model_dir=None, batch_size=32):
    """BGE cosine between the person's background (as a query) and the average embedding of each role's listings."""
    from sentence_transformers import SentenceTransformer
    from fetch.paths import PRETRAINED_DIR
    model = SentenceTransformer(str(model_dir or PRETRAINED_DIR / "bge-base-en-v1.5"), device="cpu")
    model.max_seq_length = 512
    centroids = []
    for role in ROLES:
        texts = listings.loc[listings.role.eq(role), "text"].head(n_per_role).tolist()
        if not texts:
            centroids.append(np.zeros(model.get_sentence_embedding_dimension()))
            continue
        emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        c = emb.mean(axis=0)
        centroids.append(c / (np.linalg.norm(c) or 1))
    prefix = "Represent this sentence for searching relevant passages: "  # BGE's query prefix (app/code/bge_text.py)
    people = model.encode([prefix + clean(t) for t in people_text], batch_size=batch_size, normalize_embeddings=True,
                          show_progress_bar=False)
    return people @ np.vstack(centroids).T


def ranks(scores):
    """Each role's rank for each person (1 = best); ties broken by the role order, so results are reproducible."""
    order = np.argsort(-scores, axis=1, kind="stable")
    out = np.empty_like(order)
    np.put_along_axis(out, order, np.arange(1, scores.shape[1] + 1)[None, :].repeat(len(scores), 0), axis=1)
    return out


def rrf_scores(*score_matrices, k=RRF_K):
    """Reciprocal rank fusion: the sum of 1 / (k + rank) over the methods."""
    return sum(1.0 / (k + ranks(s)) for s in score_matrices)


def baseline_scores(n_people, train_targets):
    """The baseline: every person gets the roles in order of how common they are among the training people."""
    freq = pd.Series(train_targets).value_counts().reindex(ROLES).fillna(0).to_numpy(dtype=float)
    return np.tile(freq, (n_people, 1))


# ---------- evaluation ----------

def top_k_hits(scores, targets, exclude=None, k=3):
    """1 if the person's actual role is among the method's top k, else 0. `exclude`: a role per person to leave out of
    the candidates (the current role, for moves within tech), or None."""
    s = scores.astype(float).copy()
    if exclude is not None:
        for i, role in enumerate(exclude):
            if isinstance(role, str) and role in ROLES:
                s[i, ROLES.index(role)] = -np.inf
    r = ranks(s)
    idx = np.array([ROLES.index(t) for t in targets])
    return (r[np.arange(len(idx)), idx] <= k).astype(int)


def bootstrap_ci(hits, n_boot=2000, seed=0, level=0.95):
    """Mean of 0/1 hits with a percentile bootstrap interval over people."""
    hits = np.asarray(hits, dtype=float)
    if not len(hits):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(hits, size=(n_boot, len(hits)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return hits.mean(), lo, hi


def split_people(people, test_share=0.2, folds=5, seed=0):
    """A held-out test set (scored once, at the end) and development folds, both by person and stratified by role
    and pool, so every role and pool appears in each. Returns (dev index, test index, fold number per dev person)."""
    from sklearn.model_selection import StratifiedKFold, train_test_split
    strata = people.target_bucket.astype(str) + "|" + people.pool.astype(str)
    strata = strata.where(strata.map(strata.value_counts()) >= folds + 2, "rare")
    dev, test = train_test_split(people.index, test_size=test_share, random_state=seed, stratify=strata)
    fold = pd.Series(-1, index=dev)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for f, (_, held) in enumerate(skf.split(dev, strata.loc[dev])):
        fold.iloc[held] = f
    return dev, test, fold


def evaluate(scores, people, train_targets, ks=(3, 5)):
    """Hit rates per method and pool, with bootstrap intervals. scores: {method: people x roles}; the baseline is added
    from train_targets. Within-tech moves leave the person's current role (from_bucket) out of the candidates."""
    all_scores = {"most common role (baseline)": baseline_scores(len(people), train_targets), **scores}
    exclude = people.from_bucket.where(people.from_bucket.notna(), None).tolist()
    rows = []
    for method, s in all_scores.items():
        for k in ks:
            hits = pd.Series(top_k_hits(s, people.target_bucket.tolist(), exclude, k), index=people.index)
            for pool, h in [("all career changers", hits), *hits.groupby(people.pool)]:
                mean, lo, hi = bootstrap_ci(h.to_numpy())
                rows.append({"method": method, "k": k, "pool": pool, "people": len(h), "hit_rate": mean,
                             "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def current_role_pull(scores, people):
    """For moves within tech: how often a method would put the person's current role first if it were allowed (the
    old-field bias that word matching tends to have). Returns {method: share}."""
    within = people.from_bucket.isin(ROLES).to_numpy()
    if not within.any():
        return {}
    current = np.array([ROLES.index(r) for r in people.from_bucket[within]])
    return {m: float((np.argmax(s[within], axis=1) == current).mean()) for m, s in scores.items()}
