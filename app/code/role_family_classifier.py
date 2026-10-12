"""Role-family classifier: predict a job's role family from its text, then apply it to candidate profiles.

Labels are the source-provided O*NET occupation on each corpus listing, grouped into role families below. No
human labelling is used. Because the labels come from the data provider (not from people), accuracy here means
"agreement with the provider's occupation label", not proof that a family is right for a person.

Evaluation uses a company-held-out split (all listings of a company are on one side) so reposts of the same
posting by one employer cannot leak from train to test. Models: majority-class baseline, TF-IDF + logistic
regression (title + start of description), and logistic regression on the saved BGE job embeddings.

Candidate profiles are encoded as documents (no query prefix) with the same BGE model, locally. No provider calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import rank_candidates as rc

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root, for fetch/ (shared with analysis/)
from fetch.paths import DATA_DIR  # noqa: E402  shared with analysis/ at the top
CORPUS_DIR = DATA_DIR / "matching" / "ai_ml_corpus_v1"
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


def _load_labels(corpus_dir: Path) -> dict[str, Any]:
    ids, labels, groups, titles, descs = [], [], [], [], []
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
    return {"ids": ids, "labels": labels, "groups": groups, "titles": titles, "descs": descs}


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


def _encode_profiles(profile_paths: Sequence[Path], device: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    import bge_text as bge

    raw_profiles = rc._read_profiles(profile_paths)
    model, _, _ = rc._load_model(device)
    texts = [bge.build_bge_candidate_text(rc._to_candidate(raw), raw, model.tokenizer)[0] for raw in raw_profiles]
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return raw_profiles, vectors.astype(np.float32)


def run(corpus_dir: Path, profile_paths: Sequence[Path], out_dir: Path, device: str, tfidf_train_rows: int) -> int:
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import GroupShuffleSplit

    started = time.perf_counter()
    data = _load_labels(corpus_dir)
    manifest = json.loads((corpus_dir / "bge_embeddings_manifest.json").read_text(encoding="utf-8"))
    import hashlib
    if hashlib.sha256("\0".join(data["ids"]).encode()).hexdigest() != manifest["ordered_job_ids_sha256"]:
        raise ValueError("Corpus order no longer matches the embedding matrix.")
    embeddings = np.load(corpus_dir / "bge_embeddings.npy", mmap_mode="r")

    keep = np.array([i for i, label in enumerate(data["labels"]) if label is not None])
    y_all = np.array([data["labels"][i] for i in keep])
    groups = np.array([data["groups"][i] for i in keep])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_pos, test_pos = next(splitter.split(keep, y_all, groups))
    train_idx, test_idx = keep[train_pos], keep[test_pos]
    y_train, y_test = y_all[train_pos], y_all[test_pos]
    assert not set(groups[train_pos]) & set(groups[test_pos]), "company leaked across the split"
    classes = sorted(set(y_all))
    print(f"labelled={len(keep)} train={len(train_idx)} test={len(test_idx)} classes={len(classes)} "
          f"load={time.perf_counter() - started:.0f}s", flush=True)

    results: dict[str, Any] = {}

    def record(name: str, predicted: np.ndarray, seconds: float) -> None:
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
            "fit_and_predict_seconds": round(seconds, 1),
        }
        print(name, results[name], flush=True)

    out_dir.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    record("majority_class", dummy.predict(np.zeros((len(y_test), 1))), time.perf_counter() - began)

    began = time.perf_counter()
    rng = np.random.default_rng(0)
    sub = rng.choice(len(train_idx), size=min(tfidf_train_rows, len(train_idx)), replace=False)
    docs = lambda rows: [f"{data['titles'][i]}\n{data['descs'][i]}" for i in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
    x_train = vectorizer.fit_transform(docs(train_idx[sub]))
    tfidf_model = LogisticRegression(max_iter=300, C=5.0, n_jobs=-1).fit(x_train, y_train[sub])
    record("tfidf_logreg", tfidf_model.predict(vectorizer.transform(docs(test_idx))), time.perf_counter() - began)

    began = time.perf_counter()
    x_train_e = np.asarray(embeddings[train_idx])
    x_test_e = np.asarray(embeddings[test_idx])
    bge_model = LogisticRegression(max_iter=300, C=10.0, n_jobs=-1).fit(x_train_e, y_train)
    bge_pred = bge_model.predict(x_test_e)
    record("bge_logreg", bge_pred, time.perf_counter() - began)

    best_name = max(("tfidf_logreg", "bge_logreg"), key=lambda n: results[n]["macro_f1"])
    results["selected_model"] = best_name
    report = classification_report(y_test, bge_pred, output_dict=True, zero_division=0)
    (out_dir / "bge_logreg_per_class.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    _confusion_csv(out_dir / "bge_logreg_confusion.csv", classes, y_test, bge_pred)

    raw_profiles, profile_vectors = _encode_profiles(profile_paths, device)
    probabilities = bge_model.predict_proba(profile_vectors)
    prediction_rows = []
    for raw, probs in zip(raw_profiles, probabilities):
        order = np.argsort(-probs)[:3]
        prediction_rows.append({
            "profile_id": raw["profile_id"],
            "desired_roles": raw.get("desired_roles", []),
            "top_families": [{"family": bge_model.classes_[i], "probability": round(float(probs[i]), 4)} for i in order],
        })
    (out_dir / "profile_family_predictions.json").write_text(json.dumps(prediction_rows, indent=1), encoding="utf-8")

    (out_dir / "metrics.json").write_text(json.dumps({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(corpus_dir),
        "labels": "source O*NET occupation grouped by FAMILY_OF_OCCUPATION; unmapped occupations -> Other",
        "split": "GroupShuffleSplit by company id, 80/20, seed 0",
        "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)), "classes": classes,
        "class_counts_test": dict(Counter(y_test.tolist())),
        "tfidf_train_rows": int(len(sub)),
        "results": results,
        "limitations": [
            "Labels are the provider's occupation codes, not human judgements; accuracy = agreement with them.",
            "The corpus was built mostly from title/O*NET-targeted AI/ML, data and sales pools, so class mix is not the real job market.",
            "Job text and resume text differ in style; profile predictions are a domain-shifted application, not validated.",
            "Embedding text includes the title, so the model partly learns title-to-occupation mapping.",
        ],
    }, indent=1), encoding="utf-8")
    print(f"wrote results to {out_dir} in {time.perf_counter() - started:.0f}s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--profiles", type=Path, nargs="+",
                        default=[REPO / "code" / "ai_ml_candidate_profiles.json",
                                 DATA_DIR / "candidates" / "real_candidate_profiles.json"])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tfidf-train-rows", type=int, default=80_000)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir or DATA_DIR / "matching" / f"role_family_classifier_{stamp}"
    return run(args.corpus_dir, args.profiles, out, args.device, args.tfidf_train_rows)


if __name__ == "__main__":
    raise SystemExit(main())
