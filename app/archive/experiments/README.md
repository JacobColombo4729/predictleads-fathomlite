# Archived experiments

These scripts and tests are the exploratory work that led to the current demo (October 2026). **Nothing in the app or the final pipeline imports them.** They are kept as history and for reproducibility. They are not a source of evidence for the final product: they used synthetic profiles in other job areas (water operations, logistics, retail and similar), model-generated relevance grades, and small samples, and their results were inconclusive.

The current code is in [`../../code`](../../code) (see [APP.md](../../APP.md) and [roadmap.md](../../roadmap.md)). The experiments' saved outputs are in the git-ignored `data/matching/<experiment>/` folders.

## What was here and what it found

| Experiment | Files | Question and outcome |
|---|---|---|
| Embedding comparison and section-aware text | `embedding_comparison`, `evaluate_embedding_comparison`, `section_aware_embedding`, `run_section_aware_embedding_experiment`, `evaluate_embedding_section_experiment` | Does a MiniLM embedding that keeps requirement and duty sentences beat TF-IDF? Small and mixed. |
| BGE transition | `run_bge_transition_experiment`, `evaluate_bge_transition_openai` | TF-IDF vs section-aware MiniLM vs BGE on synthetic profiles. nDCG@10 favoured BGE, P@5 was close; inconclusive. The BGE text builders from this experiment now live in `code/bge_text.py`. |
| Hybrid RRF + cross-encoder | `run_hybrid_rrf_crossencoder_experiment` | Reciprocal-rank fusion of TF-IDF and BGE looked better than either alone; adding a cross-encoder did not improve on RRF. This is why the demo uses TF-IDF + BGE RRF. |
| Weighted RRF | `run_weighted_rrf_experiment`, `run_weighted_rrf_grader_audit`, `run_weighted_rrf_model_comparison`, `run_weighted_rrf_qualification_protocol`, `analyze_weighted_rrf_dev_strong_ceiling` | Would 2:1 or 1:2 weights beat equal weights? No alternative met the pre-set rule, so equal weight stayed. The held-out profiles were never opened. |
| Faceted matching text | `matching_text_preparation`, `run_matching_text_preparation_experiment`, `analyze_matching_text_errors` | Does a faceted text representation help? Tied on P@5, mixed on nDCG; diagnostic only. |
| Fresh-profile validation | `run_fresh_profile_validation`, `evaluate_fresh_profile_validation` | Re-test on new synthetic profiles. |
| Requirement-aware reranker | `requirement_aware_reranker`, `run_requirement_reranker`, `build_requirement_rerank_union`, `evaluate_requirement_rerank` | Rerank by how well each job requirement is covered. Not adopted. |
| Description coverage | `evaluate_description_coverage` | Effect of a "substantive description" eligibility rule on the first TF-IDF baseline. |
| Retail review tool | `retail_review_app` (Streamlit), `retail_review_store` | Blinded one-job-at-a-time review screen for a retail profile. Needs `pip install streamlit`. |

The bundled profile files (`embedding_comparison_profiles.json`, `matching_text_preparation_heldout_profiles.json`, `weighted_rrf_profiles.json`) are the fictional profiles those experiments used.

## Running the archived tests

From the repository root:

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s archive/experiments/tests
```

`code/job_matching_engine.py` is a small shim that loads the live engine, and the scripts' repo-root path was adjusted when they moved, so they still resolve `data/`. The scripts are frozen history and are not maintained against future changes to the live engine.
