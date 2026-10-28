"""Role-family classifier: predict a job's role family from its title (and optionally description).

Labels are the source-provided O*NET occupation on each corpus listing, grouped into role families below. No
human labelling is used. Because the labels come from the data provider (not from people), accuracy here means
"agreement with the provider's occupation label", not proof that a family is right for a person.

**Leak guard (fixed 2026-10-13).** The first version trained on the corpus's saved BGE job embeddings, whose text
starts "Job title: ... Occupation: <O*NET occupation name>." The label comes from that occupation name, so the model
could read the answer: it scored 94.7% accuracy, but with the occupation removed the same kind of model scores
roughly 60 to 70% (see docs/ml-inventory.md). Inputs here are built by `classifier_text`, which never includes the
occupation, and a test (`tests/test_role_family_classifier.py`) keeps it that way. `--leak-control` adds the
occupation back to a keyword model on purpose, as a visible check that the guard matters.

Evaluation uses company-held-out splits (all listings of a company are on one side, so reposts of one employer
cannot leak from train to test), repeated over several random splits and reported as mean and spread. Models:
majority-class baseline, TF-IDF + logistic regression, and logistic regression on BGE embeddings of the title.

Everything runs locally; no provider calls; only aggregate metrics are printed or saved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root, for fetch/ (shared with analysis/)
from fetch.paths import DATA_DIR  # noqa: E402  shared with analysis/ at the top
CORPUS_DIR = DATA_DIR / "matching" / "ai_ml_corpus_v1"
EMBEDDING_CACHE_DIR = DATA_DIR / "matching" / "role_family_classifier_cache"
EMBEDDING_RECIPE = "title_only_v1"  # bump when classifier_text or the embedding settings change
OTHER = "Other (outside these families)"

FAMILY_OF_OCCUPATION = {
    "Data Scientists": "Data Scientist",
    "Statisticians": "Statistician / Operations Research",
    "Operations Research Analysts": "Statistician / Operations Research",
    "Business Intelligence Analysts": "BI / Data Analyst",
    "Computer and Information Research Scientists": "AI / ML Research Scientist",
    "Software Developers": "Software Developer",
    "Computer Programmers": "Software Developer",
    "Web Developers": "Web Developer",
    "Computer Systems Engineers/Architects": "Systems / Cloud Engineer",
    "Network and Computer Systems Administrators": "Network / Systems Admin",
    "Computer Network Architects": "Network / Systems Admin",
    "Database Architects": "Database Architect / Admin",
    "Database Administrators": "Database Architect / Admin",
    "Information Technology Project Managers": "IT Project / Program Manager",
    "Computer and Information Systems Managers": "IT Project / Program Manager",
    "Sales Engineers": "Technical Sales / Solutions Engineer",
    "Sales Representatives, Wholesale and Manufacturing, Technical and Scientific Products": "Sales Representative",
    "Sales Representatives of Services, Except Advertising, Insurance, Financial Services, and Travel": "Sales Representative",
    "Software Quality Assurance Analysts and Testers": "QA / Test",
    "Computer User Support Specialists": "IT Support",
    "Computer Systems Analysts": "Systems Analyst",
    "Information Security Engineers": "Security",
    "Information Security Analysts": "Security",
}


def classifier_text(title: str, description: str = "", description_chars: int = 0) -> str:
    """The text a model may see for one listing. It never includes the O*NET occupation (the label's source)."""
    text = f"Job title: {title.strip()}."
    if description_chars > 0 and description:
        text += "\n" + description[:description_chars]
    return text


def _load_labels(corpus_dir: Path) -> dict[str, Any]:
    """Labels and inputs per listing. `occupations` is kept apart on purpose: only the leak-control may use it."""
    ids, labels, groups, titles, descs, occupations = [], [], [], [], [], []
    with (corpus_dir / "jobs_prepared_corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            occupation = (record.get("onet_data") or {}).get("occupation_name")
            ids.append(record["job_id"])
            labels.append(None if not occupation else FAMILY_OF_OCCUPATION.get(occupation, OTHER))
            company = record.get("company_reference") or {}
            groups.append(str(company.get("id") or record["job_id"]))
            titles.append(record.get("title_searchable") or record.get("title_normalized") or "")
            descs.append((record.get("description_text") or "")[:2500])
            occupations.append(occupation or "")
    return {"ids": ids, "labels": labels, "groups": groups, "titles": titles, "descs": descs, "occupations": occupations}


def _confusion_csv(path: Path, classes: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[index[t], index[p]] += 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true \\ predicted", *classes])
        for c, row in zip(classes, matrix):
            writer.writerow([c, *row.tolist()])


def _title_embeddings(ids: Sequence[str], titles: Sequence[str], device: str) -> np.ndarray:
    """BGE vectors of `classifier_text(title)` for each listing, cached under data/ (chunked, resumable)."""
    import bge_text
    from sentence_transformers import SentenceTransformer

    key = hashlib.sha256((EMBEDDING_RECIPE + "\0" + "\0".join(ids)).encode()).hexdigest()[:20]
    cache = EMBEDDING_CACHE_DIR / key
    cache.mkdir(parents=True, exist_ok=True)
    final = cache / "embeddings.npy"
    if final.is_file():
        return np.load(final)
    model = SentenceTransformer(str(bge_text.model_dir()), device="cpu" if device == "auto" else device,
                                local_files_only=True, trust_remote_code=False)
    model.max_seq_length = 64  # titles are short
    size = 20_000
    began = time.perf_counter()
    parts = []
    for number, start in enumerate(range(0, len(ids), size)):
        path = cache / f"chunk_{number:05d}.npy"
        if not path.is_file():
            texts = [classifier_text(t) for t in titles[start:start + size]]
            vectors = model.encode(texts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True,
                                   show_progress_bar=False)
            np.save(path, vectors.astype(np.float32))
        parts.append(np.load(path))
        print(f"  title embeddings: {min(start + size, len(ids))}/{len(ids)} ({time.perf_counter() - began:.0f}s)", flush=True)
    matrix = np.concatenate(parts)
    np.save(final, matrix)
    return matrix


def run(corpus_dir: Path, out_dir: Path, device: str, seeds: int, tfidf_train_rows: int, leak_control: bool) -> int:
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import GroupShuffleSplit

    started = time.perf_counter()
    data = _load_labels(corpus_dir)
    keep = np.array([i for i, label in enumerate(data["labels"]) if label is not None])
    y_all = np.array([data["labels"][i] for i in keep])
    groups = np.array([data["groups"][i] for i in keep])
    classes = sorted(set(y_all))
    print(f"labelled={len(keep)} classes={len(classes)} splits={seeds} load={time.perf_counter() - started:.0f}s", flush=True)

    embeddings = _title_embeddings([data["ids"][i] for i in keep], [data["titles"][i] for i in keep], device)

    def docs(rows: np.ndarray, with_description: bool, with_occupation: bool = False) -> list[str]:
        out = []
        for i in rows:
            text = classifier_text(data["titles"][i], data["descs"][i] if with_description else "", 2500 if with_description else 0)
            if with_occupation:  # leak control only: this is exactly the mistake the guard prevents
                text += "\n" + data["occupations"][i]
            out.append(text)
        return out

    splitter = GroupShuffleSplit(n_splits=seeds, test_size=0.2, random_state=0)
    per_split: dict[str, list[dict[str, float]]] = {}
    first_split_predictions: dict[str, np.ndarray] = {}
    first_split_truth: np.ndarray | None = None
    test_class_counts: dict[str, int] = {}

    def record(name: str, split_no: int, y_test: np.ndarray, predicted: np.ndarray, seconds: float) -> None:
        entry = {"accuracy": float(accuracy_score(y_test, predicted)),
                 "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
                 "seconds": round(seconds)}
        per_split.setdefault(name, []).append(entry)
        if split_no == 0:
            first_split_predictions[name] = predicted
        print(f"  split {split_no} {name}: acc={entry['accuracy']:.4f} macroF1={entry['macro_f1']:.4f} ({entry['seconds']}s)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=False)
    for split_no, (train_pos, test_pos) in enumerate(splitter.split(keep, y_all, groups)):
        assert not set(groups[train_pos]) & set(groups[test_pos]), "company leaked across the split"
        train_idx, test_idx = keep[train_pos], keep[test_pos]
        y_train, y_test = y_all[train_pos], y_all[test_pos]
        if split_no == 0:
            first_split_truth = y_test
            test_class_counts = dict(Counter(y_test.tolist()))
        began = time.perf_counter()
        dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
        record("majority_class", split_no, y_test, dummy.predict(np.zeros((len(y_test), 1))), time.perf_counter() - began)

        began = time.perf_counter()
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
        x_tr = vec.fit_transform(docs(train_idx, False))
        model = LogisticRegression(max_iter=300, C=5.0, n_jobs=-1).fit(x_tr, y_train)
        record("tfidf_title", split_no, y_test, model.predict(vec.transform(docs(test_idx, False))), time.perf_counter() - began)

        began = time.perf_counter()
        sub = np.random.default_rng(split_no).choice(len(train_idx), size=min(tfidf_train_rows, len(train_idx)), replace=False)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
        x_tr = vec.fit_transform(docs(train_idx[sub], True))
        model = LogisticRegression(max_iter=300, C=5.0, n_jobs=-1).fit(x_tr, y_train[sub])
        record("tfidf_title_description", split_no, y_test, model.predict(vec.transform(docs(test_idx, True))), time.perf_counter() - began)

        began = time.perf_counter()
        model = LogisticRegression(max_iter=300, C=10.0, n_jobs=-1).fit(embeddings[train_pos], y_train)
        record("bge_title", split_no, y_test, model.predict(embeddings[test_pos]), time.perf_counter() - began)

        if leak_control:
            began = time.perf_counter()
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
            x_tr = vec.fit_transform(docs(train_idx[sub], True, True))
            model = LogisticRegression(max_iter=300, C=5.0, n_jobs=-1).fit(x_tr, y_train[sub])
            record("CONTROL_tfidf_with_occupation_in_text", split_no, y_test,
                   model.predict(vec.transform(docs(test_idx, True, True))), time.perf_counter() - began)

    summary: dict[str, Any] = {}
    for name, entries in per_split.items():
        acc = np.array([e["accuracy"] for e in entries])
        f1 = np.array([e["macro_f1"] for e in entries])
        summary[name] = {"accuracy_mean": round(float(acc.mean()), 4), "accuracy_std": round(float(acc.std()), 4),
                         "macro_f1_mean": round(float(f1.mean()), 4), "macro_f1_std": round(float(f1.std()), 4),
                         "per_split": entries}
    candidates = [n for n in summary if n not in ("majority_class",) and not n.startswith("CONTROL")]
    best = max(candidates, key=lambda n: summary[n]["macro_f1_mean"])
    summary["selected_model"] = best
    assert first_split_truth is not None
    report = classification_report(first_split_truth, first_split_predictions[best], output_dict=True, zero_division=0)
    (out_dir / f"{best}_per_class.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    _confusion_csv(out_dir / f"{best}_confusion.csv", classes, first_split_truth, first_split_predictions[best])

    (out_dir / "metrics.json").write_text(json.dumps({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(corpus_dir),
        "labels": "source O*NET occupation grouped by FAMILY_OF_OCCUPATION; unmapped occupations -> Other",
        "inputs": "classifier_text(title) or title + first 2,500 characters of the description; the occupation name is never an input",
        "splits": f"GroupShuffleSplit by company id, 80/20, {seeds} splits (seeds 0 to {seeds - 1} of one splitter, random_state=0)",
        "labelled_rows": int(len(keep)), "classes": classes, "class_counts_test_split0": test_class_counts,
        "tfidf_title_description_train_rows": int(tfidf_train_rows),
        "results": summary,
        "limitations": [
            "Labels are the provider's occupation codes, not human judgements; accuracy = agreement with them.",
            "The provider's labels are probably assigned mostly from job titles, so a title is most of what a model needs; this is not evidence about fit.",
            "The corpus was built mostly from title/O*NET-targeted AI/ML, data and sales pools, so class mix is not the real job market.",
            "An earlier version trained on embeddings of text that included the occupation name and reported 94.7% accuracy; that figure was inflated by label leakage.",
        ],
    }, indent=1), encoding="utf-8")
    print(f"wrote results to {out_dir} in {time.perf_counter() - started:.0f}s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, default=3, help="number of company-held-out random splits")
    parser.add_argument("--tfidf-train-rows", type=int, default=60_000, help="training rows for the (slow) title+description keyword model")
    parser.add_argument("--leak-control", action="store_true", help="also run a keyword model that can see the occupation, to show the leak")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir or DATA_DIR / "matching" / f"role_family_classifier_{stamp}"
    return run(args.corpus_dir, out, args.device, args.seeds, args.tfidf_train_rows, args.leak_control)


if __name__ == "__main__":
    raise SystemExit(main())
