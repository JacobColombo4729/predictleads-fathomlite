# Priority Queue

> Top = next. Delete a task when it's done. Add new tasks to the tier where they belong.
> The reasoning behind each task is in [`PROJECT_PLAN.md`](PROJECT_PLAN.md).

---

## Now

- **Switch `.env.example` from `S3_BUCKET` to `S3_PATH`.** Claude can't read this file under the project rules, so do it by hand. It should contain three blank lines: `AWS_PROFILE=`, `AWS_REGION=`, `S3_PATH=`. The README is already updated.
- **Tell the team** to pull, rerun `pip install -r requirements.txt` (adds `pyarrow` and `zstandard`), and switch their `.env` to `S3_PATH`.
- **Fix or delete `notebooks/scratch.py`.** It still reads `S3_BUCKET`. The README says to delete the scratch test after setup.
- **Open the pull request** from `jacob/eda-and-plan` into `main` on GitHub. From now on, commit only on branches.

## Blocked

- **Team:**
  - Agree on the focus: career-transition role discovery, TF-IDF vs. embeddings. Blocks P0.
  - Choose the source background for the persona. Blocks P0.
  - Assign owners for matching, evaluation, and the explanations/card/demo.
  - Set a meeting cadence and a place for async discussion.
  - Confirm the final due date and deliverables.
- **PredictLeads:**
  - Does role discovery for career changers fit what they want?
  - Schema documentation, especially `onet_data`, and whether `normalized_title`, `categories` or `tags` come from O*NET
  - Are daily snapshots full copies or incremental changes?
  - What do `status` values mean?
  - Is the job openings file sorted in some way, for example by company or date?

## P0

- **Record the agreed focus and source background** in the plan's decision log.
- **Finish the feasibility checks** in the EDA notebook:
  - **Is the sample representative?** It's the first 20,000 documents, not a random draw, and covers only 10,907 companies. Check how postings spread across companies and `first_seen_at` dates. If the file is sorted, streaming with random sampling (P3) moves up to P0.
  - Number of persona postings in the sample (entry-level, plausible target families). This also decides whether streaming the full file moves up.
  - What `contract_types` and `location_data` contain, and which values define the persona
  - `status` values, and how many postings are still live based on `last_seen_at`
  - Share of postings in English (`attributes.language`)
  - Description lengths (fill rate already known: 82%). This decides whether embeddings need chunking.
  - Number of O*NET families present
  - What `tags` and `categories` contain: skills, labels, or derived from O*NET?
  - Share of job-opening companies found in the companies table
- **Check whether `normalized_title`, `categories` and `tags` are derived from O*NET.** Keep any that are out of the matching text.
- **Write the hypothesis and success threshold** before running any model. A draft is in the plan.
- **Build the corpus.**
  - Live, English postings with descriptions, filtered by the persona's seniority, contract type and region
  - Parse `first_seen_at`, `last_seen_at` and `posted_at` to datetimes
  - Clean the text: strip HTML, collapse whitespace, remove boilerplate sections where headers make them detectable
  - Remove duplicate postings: same company, same title text, near-identical description
  - Keep `onet_data` out of the matching text
  - Save the list of posting IDs. Every method, including title keyword search, ranks this same set.
- **Write the profiles.**
  - Remove personal details from team résumés: names, contact details, employers, schools, locations.
  - Write synthetic profiles covering different backgrounds, for 10–20 in total.
  - Use one template for all profiles: summary + experience bullets + skills.
  - Assign each profile its obvious O*NET families **before any run**.
  - Split them: about a third for dev, the rest for test.
  - Store them in `data/`. Never commit them.
- **Write the rating rubric** for graded relevance (0 = not relevant, 1 = stretch, 2 = strong match), with one worked example per grade. Agree on it before rating.
- **Install the embedding library.** `pip install sentence-transformers`, pin it in `requirements.txt` and tell the team.
- **Build the baselines.**
  - Random ranking (sanity floor)
  - Title keyword search on `title` (100% filled), not `normalized_title`
  - TF-IDF + cosine, using the plan's settings, fit on postings only
- **Build the challenger: embeddings, run locally.**
  - Compare `all-MiniLM-L6-v2` with one stronger small retrieval model (`bge-small-en-v1.5` or `e5-small-v2`) on dev profiles. Pin the winner's name and version.
  - Chunk long descriptions and score each posting by its best chunk. Compare with plain truncation.
  - Use each model's query/passage prefixes, and normalize the vectors.
  - Cache embeddings in `data/`, keyed by posting ID.
- **Build the hybrid:** reciprocal rank fusion of TF-IDF and embeddings.
- **Tune all methods equally hard,** on dev profiles only.
- **Label relevance.**
  - Automatic O*NET discovery labels
  - Blind, graded team ratings of the pooled top 10 from every method, shuffled with the method hidden
  - Two raters on the same ~20% of items. Compute Cohen's kappa. If it's low, refine the rubric and rate again.
- **Run the error analysis.** This is the core finding.
  - Unique finds per method, with an explanation of why the others missed them
  - Bad results in each method's top 10
  - Failures grouped by type, with counts: vocabulary mismatch, boilerplate, seniority/experience mismatch, title-word-only match, duplicates

## P1

- **Run the test profiles once,** with all settings fixed.
- **Compute the metrics at k = 10:** precision, nDCG, discovery rate, family diversity, unique finds.
- **Statistics:**
  - Per-profile metrics with mean and 95% paired bootstrap confidence intervals
  - A paired permutation test for each method vs. embeddings
  - Benjamini–Hochberg correction across those comparisons
- **Reproducibility:** fixed seeds, the embedding model pinned, and each run's settings recorded with its results.
- **Grounded match explanations:** shared high-weight TF-IDF terms, plus the most similar sentence pairs from the embeddings.
- **Flag "N+ years" experience requirements** with a regular expression. They're the most common blocker for career changers.
- **Choose the demo format:** a notebook or a small Streamlit app. If Streamlit, pin it and tell the team.
- **Minimal recommendation card:**
  - Why it matches
  - Experience flag
  - Company name and domain from the job opening's `included` objects; location from the companies table, joined through `relationships.company.data.id`
  - "First seen N days ago" from `first_seen_at`
  - Link to the original posting (`url`)

## P2

- **Run 2–3 ablations,** on dev profiles only:
  - **LSA** (`TruncatedSVD` on TF-IDF). The most informative one.
  - BM25 vs. TF-IDF cosine (`pip install rank-bm25`; pin it and tell the team)
  - TF-IDF with and without bigrams
  - Embedding the title only, the description only, or both
  - With and without the requirements section
- **Extract required and preferred requirements, rules first.**
  - Split descriptions on section headers ("Requirements", "Qualifications", "Preferred", "Nice to have", "Bonus").
  - Measure how often the rules work on dev postings.
  - Consider GenAI only for what rules can't handle, and only on synthetic postings unless the advisors approve it in writing.
  - If `tags` turns out to hold skills, use it as structured input.
- **Gap analysis:** compare each profile against the required and preferred lists.
- **Grounding check:** verify automatically that every quoted phrase appears word for word in the profile or posting. Report the pass rate.
- **Complete the card:**
  - Gaps, marked required or preferred
  - Salary where available (about 43% filled), converted to one time unit using `salary_time_unit`
  - Company description, where the company is in the companies table
- **Write up the evaluation:** hypothesis, method, results with confidence intervals, error analysis with failure-type counts, the rater-agreement score, and limitations, including the small number of profiles.

## P3

- **Polish the pitch.** Each team member is pitched for a role the system recommended.
- **Make a UMAP plot** of the embedded postings, colored by O*NET family, with profiles marked. `pip install umap-learn`, pin it and tell the team. Use it for the picture only; never feed it into the matching.
- **Move the shared loaders into `src/`:** `fetch`, `stream_jsonl`, `load_cached`, `unpack`. Replace `DATA_DIR = Path("../data")` with a path based on the project root, because the relative path breaks outside `jacob_eda/`.
- **Fix the README's inconsistent names.** The kernel is registered as `project_leads`, but the text says "Python (team_project)". The README also says `.venv` while your venv is `PL`.
- **Stream the full 19 GB job openings file** with filters and random sampling. Only needed if P0's feasibility checks find too few persona postings or a biased sample; in either case, move it up to P0.

## Always

- Clear notebook outputs before every commit.
- Never commit data, résumés, Parquet files, or `.env`.
- After a kernel restart, run the notebook from the top.
- If the notebook looks out of date in VS Code, run **File: Revert File**. Don't save the older copy over newer changes.
- Record team decisions in the plan's decision log.
- When you add a package: pin it in `requirements.txt`, commit, and tell the team.
- Keep track of every local copy of the data, for the final deletion.

## Finish

- Permanently delete every copy of the data: `data/` (Parquet and `.jsonl.zst` files), downloads, résumés, and any S3 copies.
- Confirm the deletion in writing if asked.
