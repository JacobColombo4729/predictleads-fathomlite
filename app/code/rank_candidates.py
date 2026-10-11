"""One command path for the AI/ML career-transition direction.

    python code/rank_candidates.py build-corpus --targeted-dir data/matching/targeted_pool_<ts>
    python code/rank_candidates.py encode --corpus-dir data/matching/ai_ml_corpus_v1
    python code/rank_candidates.py rank --corpus-dir ... --profiles code/ai_ml_candidate_profiles.json \
        data/candidates/real_candidate_profiles.json

Methods per profile: tfidf_cosine, bge_cosine, rrf_equal (reciprocal-rank fusion of the two, k=60).
Scores are uncalibrated retrieval signals. Reuses the existing TF-IDF engine, BGE job/candidate
text builders and hard filters unchanged. Duplicate listings are collapsed to one representative
for ranking (see duplicate_groups.jsonl); nothing is deleted from the source files.
No provider (paid API) call is made anywhere in this file.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent  # data/, models/ and .env are shared with analysis/ at the top
GPU_LIBS = REPO_ROOT / "data" / "_gpu_libs"
if GPU_LIBS.is_dir() and "--cpu" not in sys.argv:
    sys.path.insert(0, str(GPU_LIBS))  # CUDA build of the same torch version, installed outside the venv
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import job_matching_engine as engine  # noqa: E402
import bge_text as bge  # noqa: E402

SAMPLE_PATH = REPO_ROOT / "data/discovery/role_exploration/roles_corrected_20261006T000000Z/jobs_prepared_random_sample.jsonl"
BGE_SNAPSHOT = bge.model_dir()  # models/bge-base-en-v1.5, or $PREDICTLEADS_BGE_MODEL_DIR
DEFAULT_CORPUS_DIR = REPO_ROOT / "data/matching/ai_ml_corpus_v1"
DEPTH = 50
RRF_K = 60
TOP_K = 10
METHODS = ("tfidf_cosine", "bge_cosine", "rrf_equal")
SCORE_TYPES = {"tfidf_cosine": "tfidf_cosine", "bge_cosine": "bge_cosine_similarity", "rrf_equal": "reciprocal_rank_fusion"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _dup_key(record: dict[str, Any]) -> str:
    company = (record.get("company_reference") or {}).get("id") or ""
    title = _norm(str(record.get("title_searchable") or ""))
    description = _norm(str(record.get("description_text") or ""))
    return hashlib.sha1("\x1f".join((str(company), title, description)).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_corpus(targeted_dir: Path, out_dir: Path, sample_limit: int | None = None) -> int:
    pool_path = targeted_dir / "targeted_pool.jsonl"
    summary = json.loads((targeted_dir / "scan_summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete_file_scanned"):
        print("WARNING: the targeted scan did not reach EOF; the corpus will be partial.", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=False)
    seen_keys: dict[str, str] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    counts: Counter[str] = Counter()
    sample_ids: set[str] = set()
    with SAMPLE_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if sample_limit is not None and line_number >= sample_limit:
                break
            sample_ids.add(json.loads(line)["job_id"])
    corpus_path = out_dir / "jobs_prepared_corpus.jsonl"
    with corpus_path.open("x", encoding="utf-8", newline="\n") as out:
        def consider(record: dict[str, Any], sources: list[str]) -> None:
            job_id = str(record.get("job_id"))
            counts["records_seen_" + sources[0]] += 1
            if job_id in seen_ids:
                counts["same_job_id_in_both_sources"] += 1
                return
            seen_ids.add(job_id)
            if not (record.get("description_usable") and str(record.get("description_text") or "").strip()):
                counts["excluded_no_usable_description"] += 1
                return
            key = _dup_key(record)
            if key in seen_keys:
                counts["collapsed_duplicate_listing"] += 1
                groups[key].append(job_id)
                return
            seen_keys[key] = job_id
            groups[key] = [job_id]
            record["corpus_sources"] = sources
            record["duplicate_key"] = key
            record.pop("description_raw", None)  # memory: raw HTML stays in targeted_pool.jsonl / the random sample
            record["description_raw_omitted_from_corpus"] = True
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts["representatives_written"] += 1
            for source in sources:
                counts["representatives_from_" + source] += 1

        with pool_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                consider(record, ["targeted_pool"] + (["random_sample"] if record.get("job_id") in sample_ids else []))
        with SAMPLE_PATH.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if sample_limit is not None and line_number >= sample_limit:
                    break
                consider(json.loads(line), ["random_sample"])
    with (out_dir / "duplicate_groups.jsonl").open("x", encoding="utf-8", newline="\n") as dup:
        group_sizes: Counter[int] = Counter()
        for key, members in groups.items():
            group_sizes[min(len(members), 10)] += 1
            if len(members) > 1:
                dup.write(json.dumps({"duplicate_key": key, "representative_job_id": members[0],
                                      "listing_count": len(members), "job_ids": members[:50]}) + "\n")
    report = {
        "corpus_version": "ai_ml_corpus_v1", "targeted_pool_dir": str(targeted_dir),
        "targeted_pool_sha256": _sha256_file(pool_path), "random_sample_sha256": _sha256_file(SAMPLE_PATH),
        "targeted_scan_complete": summary.get("complete_file_scanned"),
        "counts": dict(counts), "duplicate_group_size_histogram_capped_at_10": dict(sorted(group_sizes.items())),
        "duplicate_rule": "same company reference id + same lowercased title + same whitespace-normalized description text; first listing kept, others recorded in duplicate_groups.jsonl",
        "limitations": ["description_raw is omitted from the corpus file (kept in the source pool/sample files); description_text is unchanged.",
                        "Near-duplicate reposts with edited text are not collapsed.",
                        "Records without a usable description are excluded from this ranking corpus (they stay in the source files).",
                        "Historical snapshot; availability is not verified."],
    }
    (out_dir / "corpus_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


def _load_model(device_choice: str):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if device_choice != "cpu" and torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(BGE_SNAPSHOT), device=device, local_files_only=True, trust_remote_code=False)
    model.max_seq_length = bge.MAX_BGE_TOKENS
    return model, device, torch.__version__


def encode(corpus_dir: Path, device_choice: str, batch_size: int) -> int:
    jobs, load_report = engine.load_prepared_jobs(corpus_dir / "jobs_prepared_corpus.jsonl")
    model, device, torch_version = _load_model(device_choice)
    print(f"jobs={len(jobs)} device={device} torch={torch_version}", flush=True)
    started = time.perf_counter()
    texts = [bge.build_bge_job_text(job.source, model.tokenizer)[0] for job in jobs]
    prep_seconds = time.perf_counter() - started
    print(f"job texts built in {prep_seconds:.0f}s", flush=True)
    chunk_dir = corpus_dir / "bge_chunks"
    chunk_dir.mkdir(exist_ok=True)
    size = 4096
    encode_started = time.perf_counter()
    for number, start in enumerate(range(0, len(texts), size)):
        path = chunk_dir / f"chunk_{number:05d}.npy"
        if path.exists():
            continue
        vectors = model.encode(texts[start:start + size], batch_size=batch_size, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
        np.save(path, vectors.astype(np.float32))
        done = min(start + size, len(texts))
        print(f"encoded {done}/{len(texts)} elapsed={time.perf_counter() - encode_started:.0f}s", flush=True)
    ids_hash = hashlib.sha256("\0".join(j.job_id for j in jobs).encode()).hexdigest()
    texts_hash = hashlib.sha256("\0".join(texts).encode()).hexdigest()
    row_count, load_summary = len(jobs), load_report.to_dict()
    del texts, jobs  # release job text/records before the merge (the first merge ran out of memory with them held)
    gc.collect()
    manifest = _merge_chunks(corpus_dir, row_count, ids_hash)
    manifest.update({
        "device": device, "torch_version": torch_version, "job_texts_sha256": texts_hash,
        "text_prep_seconds": prep_seconds, "encode_seconds": time.perf_counter() - encode_started,
        "load_report": load_summary,
    })
    (corpus_dir / "bge_embeddings_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "device", "encode_seconds")}), flush=True)
    return 0


def _merge_chunks(corpus_dir: Path, expected_rows: int, ids_hash: str) -> dict[str, Any]:
    """Copy validated chunk files into one .npy through a memory-mapped output (bounded memory)."""
    chunk_paths = sorted((corpus_dir / "bge_chunks").glob("chunk_*.npy"))
    shapes = []
    for number, path in enumerate(chunk_paths):
        if path.name != f"chunk_{number:05d}.npy":
            raise RuntimeError(f"Missing or unexpected chunk file at position {number}: {path.name}")
        shapes.append(np.load(path, mmap_mode="r").shape)
    rows = sum(shape[0] for shape in shapes)
    dims = {shape[1] for shape in shapes}
    if rows != expected_rows or len(dims) != 1:
        raise RuntimeError(f"Chunks cover {rows} rows (dims {dims}); expected {expected_rows}.")
    final_path = corpus_dir / "bge_embeddings.npy"
    temp_path = corpus_dir / "bge_embeddings.partial.npy"
    out = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float32, shape=(rows, dims.pop()))
    start = 0
    for path in chunk_paths:
        block = np.load(path, mmap_mode="r")
        out[start:start + block.shape[0]] = block
        start += block.shape[0]
    out.flush()
    del out
    temp_path.replace(final_path)
    return {
        "model_id": bge.BGE_ID, "model_revision": bge.BGE_REVISION, "representation_id": bge.BGE_REPRESENTATION,
        "rows": rows, "dimension": int(np.load(final_path, mmap_mode="r").shape[1]), "normalized": True,
        "max_sequence_length_tokens": bge.MAX_BGE_TOKENS, "ordered_job_ids_sha256": ids_hash,
        "corpus_sha256": _sha256_file(corpus_dir / "jobs_prepared_corpus.jsonl"),
        "embedding_file_sha256": _sha256_file(final_path), "chunk_count": len(chunk_paths),
        "note": "Same BGE model revision and job-text builder as the earlier reference cache; encoded in fp32.",
    }


def finalize(corpus_dir: Path) -> int:
    """Merge already-encoded, validated chunks without rebuilding job text or re-encoding anything."""
    ids = []
    with (corpus_dir / "jobs_prepared_corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            ids.append(json.loads(line)["job_id"])
    manifest = _merge_chunks(corpus_dir, len(ids), hashlib.sha256("\0".join(ids).encode()).hexdigest())
    verification = {}
    for name in ("chunk_verification_structure.json", "chunk_verification_spotcheck.json"):
        path = corpus_dir / name
        if path.is_file():
            verification[name] = json.loads(path.read_text(encoding="utf-8"))
    manifest.update({
        "job_texts_sha256": None,
        "finalized_from_saved_chunks": True,
        "finalize_note": "Merged after the first encode run exhausted memory at the merge step. Job texts were not rebuilt, so their hash was not recorded. Row order is tied to the corpus by ordered_job_ids_sha256; vectors were spot-checked by re-encoding 60 sampled jobs (see verification).",
        "verification": verification,
    })
    (corpus_dir / "bge_embeddings_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "dimension", "embedding_file_sha256")}), flush=True)
    return 0


def export_serving(corpus_dir: Path, out_dir: Path) -> int:
    """Write the demo-serving subset: every listing whose source status is not explicitly closed.

    Same records, same vectors (row subset), same order as the full corpus. Used by the backend so it
    starts faster and holds less memory. Unknown status stays labelled unknown; nothing is inferred.
    """
    manifest = json.loads((corpus_dir / "bge_embeddings_manifest.json").read_text(encoding="utf-8"))
    matrix = np.load(corpus_dir / "bge_embeddings.npy", mmap_mode="r")
    out_dir.mkdir(parents=True, exist_ok=False)
    keep: list[int] = []
    ids: list[str] = []
    source_path = corpus_dir / "jobs_prepared_corpus.jsonl"
    target_path = out_dir / "jobs_prepared_corpus.jsonl"
    with source_path.open(encoding="utf-8") as source, target_path.open("x", encoding="utf-8", newline="\n") as target:
        for index, line in enumerate(source):
            record = json.loads(line)
            if str(record.get("status_derived") or "unknown").casefold() == "closed":
                continue
            target.write(line)
            keep.append(index)
            ids.append(str(record["job_id"]))
    if matrix.shape[0] != index + 1:
        raise RuntimeError("Corpus rows and embedding rows differ.")
    subset = np.lib.format.open_memmap(out_dir / "bge_embeddings.npy", mode="w+", dtype=np.float32,
                                       shape=(len(keep), matrix.shape[1]))
    for start in range(0, len(keep), 8192):
        subset[start:start + 8192] = matrix[keep[start:start + 8192]]
    subset.flush()
    del subset
    out_manifest = dict(manifest)
    out_manifest.update({
        "rows": len(keep), "ordered_job_ids_sha256": hashlib.sha256("\0".join(ids).encode()).hexdigest(),
        "corpus_sha256": _sha256_file(out_dir / "jobs_prepared_corpus.jsonl"),
        "embedding_file_sha256": _sha256_file(out_dir / "bge_embeddings.npy"),
        "serving_subset_of": str(corpus_dir), "serving_rule": "status_derived != closed",
        "source_embedding_file_sha256": manifest.get("embedding_file_sha256"),
    })
    out_manifest.pop("verification", None)
    (out_dir / "bge_embeddings_manifest.json").write_text(json.dumps(out_manifest, indent=1), encoding="utf-8")
    print(json.dumps({"serving_rows": len(keep), "dimension": int(matrix.shape[1])}))
    return 0


def _read_profiles(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(data["profiles"] if isinstance(data, dict) else data)
    ids = [row["profile_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate profile_id across profile files.")
    return rows


def _to_candidate(row: dict[str, Any]) -> engine.CandidateProfile:
    hard = row.get("hard_constraints", {})
    preferences = {
        "exclude_explicitly_closed": hard.get("exclude_explicitly_closed") is True,
        "require_location_match": hard.get("require_location_match") is True,
        "keywords": list(row.get("soft_preferences", [])),
        "work_arrangement": list(row.get("work_arrangement_preferences", [])),
    }
    return engine.CandidateProfile(
        profile_id=str(row["profile_id"]),
        experience=tuple([*row["work_history"], *row.get("accomplishments", []), *row.get("education_certifications", [])]),
        evidenced_skills=tuple(f"{s['skill']}: {s['evidence']}" for s in row["skills_with_evidence"]),
        desired_roles=tuple(row.get("desired_roles", [])), location=row.get("location"), preferences=preferences)


def rank(corpus_dir: Path, profile_paths: Sequence[Path], out_dir: Path, device_choice: str) -> int:
    manifest = json.loads((corpus_dir / "bge_embeddings_manifest.json").read_text(encoding="utf-8"))
    jobs, _ = engine.load_prepared_jobs(corpus_dir / "jobs_prepared_corpus.jsonl")
    if hashlib.sha256("\0".join(j.job_id for j in jobs).encode()).hexdigest() != manifest["ordered_job_ids_sha256"]:
        raise ValueError("Corpus order no longer matches the embedding matrix.")
    matrix = np.load(corpus_dir / "bge_embeddings.npy", mmap_mode="r")
    model, device, _ = _load_model(device_choice)
    started = time.perf_counter()
    corpus = engine.fit_tfidf(jobs)
    print(f"TF-IDF fit {time.perf_counter() - started:.0f}s on {len(jobs)} jobs", flush=True)
    raw_profiles = _read_profiles(profile_paths)
    out_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    deep: list[dict[str, Any]] = []
    eligibility: dict[str, int] = {}
    all_indices = list(range(len(jobs)))
    for raw in raw_profiles:
        profile = _to_candidate(raw)
        eligible = bge.eligible_indices(corpus, profile, all_indices)
        eligibility[profile.profile_id] = len(eligible)
        tfidf_rows = engine.rank_tfidf(corpus, profile, DEPTH, candidate_indices=eligible, condition="ai_ml_corpus_v1")
        query_text, _ = bge.build_bge_candidate_text(profile, raw, model.tokenizer)
        query = model.encode([bge.BGE_QUERY_PREFIX + query_text], convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False)[0]
        scores = np.asarray(matrix[eligible]) @ query
        order = sorted(zip(scores.tolist(), eligible), key=lambda p: (-p[0], jobs[p[1]].job_id))[:DEPTH]
        bge_rows = [{"job_id": jobs[i].job_id, "rank": r, "score": s} for r, (s, i) in enumerate(order, 1)]
        fused: dict[str, float] = defaultdict(float)
        for source in (tfidf_rows, bge_rows):
            for item in source:
                fused[str(item["job_id"])] += 1.0 / (RRF_K + int(item["rank"]))
        by_id = {j.job_id: j for j in jobs}
        method_rows = {
            "tfidf_cosine": [(r["job_id"], r["score"]) for r in tfidf_rows[:TOP_K]],
            "bge_cosine": [(r["job_id"], r["score"]) for r in bge_rows[:TOP_K]],
            "rrf_equal": sorted(fused.items(), key=lambda p: (-p[1], p[0]))[:TOP_K],
        }
        deep_rows = {
            "tfidf_cosine": [(r["job_id"], r["score"]) for r in tfidf_rows],
            "bge_cosine": [(r["job_id"], r["score"]) for r in bge_rows],
            "rrf_equal": sorted(fused.items(), key=lambda p: (-p[1], p[0]))[:DEPTH],
        }
        for method in METHODS:
            for rank_number, (job_id, score) in enumerate(deep_rows[method], 1):
                deep.append({"profile_id": profile.profile_id, "method": method, "rank": rank_number,
                             "job_id": job_id, "title": by_id[str(job_id)].searchable_title, "score": float(score)})
        for method in METHODS:
            for rank_number, (job_id, score) in enumerate(method_rows[method], 1):
                job = by_id[str(job_id)]
                rows.append({"profile_id": profile.profile_id, "method": method, "rank": rank_number,
                             "job_id": job_id, "title": job.searchable_title, "score": float(score),
                             "score_type": SCORE_TYPES[method],
                             "corpus_sources": "+".join(job.source.get("corpus_sources", []))})
    with (out_dir / "rankings.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "rankings_top50.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deep[0]))
        writer.writeheader()
        writer.writerows(deep)
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "corpus_dir": str(corpus_dir), "corpus_jobs": len(jobs), "embedding_manifest": manifest,
        "profiles": [r["profile_id"] for r in raw_profiles],
        "profile_file_sha256": {str(p): _sha256_file(p) for p in profile_paths},
        "eligible_jobs_after_hard_filters": eligibility,
        "settings": {"depth": DEPTH, "rrf_k": RRF_K, "top_k": TOP_K, "rrf_weights": "1:1", "query_device": device},
        "note": "No provider calls. Scores are uncalibrated retrieval signals, not match percentages.",
    }, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} ranking rows for {len(raw_profiles)} profiles to {out_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu", action="store_true", help="force CPU and the project's own torch build")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-corpus")
    b.add_argument("--targeted-dir", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    b.add_argument("--sample-limit", type=int, default=None, help="testing only: use the first N random-sample rows")
    e = sub.add_parser("encode")
    e.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    e.add_argument("--batch-size", type=int, default=32)
    x = sub.add_parser("export-serving", help="write the not-closed subset used by the demo backend")
    x.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    x.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/matching/ai_ml_serving_v1")
    f = sub.add_parser("finalize-encode", help="merge saved chunks only (no re-encoding)")
    f.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    r = sub.add_parser("rank")
    r.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    r.add_argument("--profiles", type=Path, nargs="+", required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    device = "cpu" if args.cpu else "auto"
    if args.command == "build-corpus":
        return build_corpus(args.targeted_dir, args.out_dir, args.sample_limit)
    if args.command == "encode":
        return encode(args.corpus_dir, device, args.batch_size)
    if args.command == "export-serving":
        return export_serving(args.corpus_dir, args.out_dir)
    if args.command == "finalize-encode":
        return finalize(args.corpus_dir)
    return rank(args.corpus_dir, args.profiles, args.out_dir, device)


if __name__ == "__main__":
    raise SystemExit(main())
