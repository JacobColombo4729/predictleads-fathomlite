# Project Plan

PredictLeads × TripleTen externship project.

**Status:** Draft for team discussion · **Last updated:** 2026-10-06

> This is a living document. Propose changes in a pull request, and record agreed outcomes in the [Decision log](#8-decision-log).
> Items marked **TBD** still need a team decision or input from PredictLeads or our advisors.
> Follow the [Data & Security Rules](README.md#data--security-rules): no data values, `df.head()` output or screenshots in this file.

---

## Contents

1. [Goals](#1-goals)
2. [Data](#2-data)
3. [Current status](#3-current-status)
4. [Three-week plan](#4-three-week-plan)
5. [Evaluation](#5-evaluation)
6. [Roles](#6-roles)
7. [Scope](#7-scope)
8. [Decision log](#8-decision-log)
9. [Risks and compliance](#9-risks-and-compliance)
10. [Open questions](#10-open-questions)
11. [Discussion notes](#11-discussion-notes)

---

## 1. Goals

**Proposed. Needs team agreement.** This merges two ideas: the overall goal of finding the optimal job search strategy with a narrow focus and maximum depth, and a teammate's proposal for career-transition role discovery.

| | |
|---|---|
| **Overall goal** | Find the optimal job search strategy |
| **Focused question** | For a career changer from **one specific background**, which roles they wouldn't think to search for are a realistic match? And does **semantic matching find them better than keyword matching**? |
| **Persona** | Career changers from one source background. TBD: pick one that matches at least one teammate's résumé, for example customer service. Target postings are filtered by `seniority`, `contract_types` and region (`location_data`). |
| **Search decision** | **What to apply for**: discovering roles beyond the obvious job titles |
| **Where the depth comes from** | A measured comparison of TF-IDF and embeddings, plus grounded explanations of each match and its skill gaps |
| **Final deliverables** | A recommendation demo, an evaluation write-up, and a final pitch |
| **Timebox** | Three weeks |
| **Final due date** | TBD |

### The recommendation card

For each recommended posting, the platform shows:

1. **Why it matches:** which parts of the profile's experience fit the posting.
2. **Gaps:** requirements the profile doesn't show, each marked as **required** or **preferred**.
3. **Company information:**
   - Name and domain, from the job opening's `included` objects. Available for every posting.
   - Description and location, from the companies data, where the company is found there.
4. **Posting details:**
   - Salary where available, shown with its time unit (`salary_time_unit`)
   - How recently it was first seen (`first_seen_at`)
   - A link to the original posting (`url`)

### Success criteria

- Embeddings beat TF-IDF on the [evaluation](#5-evaluation) metrics. Or, if they don't, we can explain why with evidence.
- We show concrete examples of relevant **cross-family** roles that embeddings find and TF-IDF misses.
- Every explanation and gap claim traces back to text in the profile or the job description.

---

## 2. Data

Daily snapshots in S3, laid out as `YYYY/MM/DD/`. We are currently working from the **2026-09-30** snapshot.

| Dataset | S3 file | Compressed size | Notes |
|---|---|---|---|
| Companies | `companies_YYYY_MM_DD.jsonl.zst` | ~252 MB | Small enough to load fully |
| Job openings | `job_openings_YYYY_MM_DD.jsonl.zst` | ~19 GB | Too big for memory; filter while streaming |

**Format.** Each line is a JSON:API document with two keys:

- `data`: a list of the actual records, usually with `id`, `type`, `attributes` and `relationships`.
- `included`: a list of related objects that the records reference, such as the company behind a job opening. The same object repeats across documents, so remove duplicates by `(type, id)`.

**Local cache.** The loader in `jacob_eda/jacob_eda.ipynb` saves each dataset as Parquet in `data/` (git-ignored) after the first read. Later runs load the Parquet files in seconds instead of parsing the JSON for about 10 minutes.

### Fields the project depends on

| Need | Fields |
|---|---|
| Matching text | `attributes.description` (82% filled), `attributes.title` (100%). **Never** `onet_data.*`; see [leakage rules](#leakage-rules). |
| Title keyword search baseline | `attributes.title` (100%). `normalized_title` is only 91% filled and may be derived from O*NET. |
| Language filter | `attributes.language` (95%) |
| Persona filters | `attributes.seniority` (100%), `attributes.contract_types` (100%), `attributes.location_data` (100%) for region |
| Open-posting filter | `attributes.status` (65%), `attributes.last_seen_at` (100%) |
| Possible skills for gap analysis | `attributes.tags` (100%). Contents still to check. |
| Evaluation labels | `attributes.onet_data.code`, `.family`, `.occupation_name` (99.9%). `attributes.categories` (100%) is a possible second check. |
| Card: posting details | `attributes.salary_data.*` (43%) with `salary_time_unit`, `attributes.first_seen_at` (100%), `attributes.url` (100%) |
| Card: company information | Name and domain from `job_openings_included` (99.6% / 100%). Description (90%) and location (94%) from the companies table, joined on `relationships.company.data.id` = company `id`. |

### Data quirks

- **The samples are the *first* N documents of the file, not random samples.** If the file is sorted, for example by company or date, the sample is biased. For example, the first 20,000 job openings come from only 10,907 companies. Check this before building the matching set from the sample.
- **Dates are stored as strings.** Parse `first_seen_at`, `last_seen_at` and `posted_at` with `pd.to_datetime` before use.
- **Salaries mix time units** (`salary_time_unit`). Convert to one unit before showing or comparing them.
- **Some postings may be closed.** `status` is 35% empty. Use `last_seen_at` relative to the snapshot date to judge whether a posting is still live.
- **Not every job's company may be in the companies file.** Measure the match rate before relying on company descriptions.

### Schemas

Like `df.info()` for each flattened table: column names, non-null counts and dtypes only, never data values. The schema cell in `jacob_eda/jacob_eda.ipynb` generates this section. Rerun that cell to refresh it; don't edit it by hand.

<!-- schema:start -->
_Generated 2026-10-06 08:17 from the 2026-09-30 snapshot._

#### `companies`: 340,669 rows × 23 columns

| # | Column | Non-null | Null % | Dtype |
|---|---|---|---|---|
| 0 | `id` | 340,669 | 0.0% | object |
| 1 | `type` | 340,669 | 0.0% | object |
| 2 | `attributes.domain` | 340,669 | 0.0% | object |
| 3 | `attributes.company_name` | 338,499 | 0.6% | object |
| 4 | `attributes.friendly_company_name` | 338,521 | 0.6% | object |
| 5 | `attributes.ticker` | 5,770 | 98.3% | object |
| 6 | `attributes.meta_title` | 339,031 | 0.5% | object |
| 7 | `attributes.meta_description` | 302,619 | 11.2% | object |
| 8 | `attributes.description` | 307,849 | 9.6% | object |
| 9 | `attributes.description_short` | 307,849 | 9.6% | object |
| 10 | `attributes.language` | 337,961 | 0.8% | object |
| 11 | `attributes.location` | 321,122 | 5.7% | object |
| 12 | `attributes.location_data` | 340,669 | 0.0% | object |
| 13 | `attributes.last_extraction_states` | 340,669 | 0.0% | object |
| 14 | `relationships.redirects_to.data` | 0 | 100.0% | float64 |
| 15 | `relationships.redirects_to.meta.reason` | 18 | 100.0% | object |
| 16 | `relationships.parent_company.data` | 0 | 100.0% | float64 |
| 17 | `relationships.subsidiary_companies.data` | 340,669 | 0.0% | object |
| 18 | `relationships.similar_companies.data` | 340,669 | 0.0% | object |
| 19 | `relationships.parent_company.data.id` | 25,287 | 92.6% | object |
| 20 | `relationships.parent_company.data.type` | 25,287 | 92.6% | object |
| 21 | `relationships.redirects_to.data.id` | 2,013 | 99.4% | object |
| 22 | `relationships.redirects_to.data.type` | 2,013 | 99.4% | object |

#### `companies_included`: 706,570 rows × 5 columns (duplicates removed by type and id)

| # | Column | Non-null | Null % | Dtype |
|---|---|---|---|---|
| 0 | `id` | 706,570 | 0.0% | object |
| 1 | `type` | 706,570 | 0.0% | object |
| 2 | `attributes.domain` | 706,570 | 0.0% | object |
| 3 | `attributes.company_name` | 682,429 | 3.4% | object |
| 4 | `attributes.ticker` | 8,300 | 98.8% | object |

#### `job_openings`: 20,000 rows × 34 columns (from the first 20,000 job-opening documents)

| # | Column | Non-null | Null % | Dtype |
|---|---|---|---|---|
| 0 | `id` | 20,000 | 0.0% | object |
| 1 | `type` | 20,000 | 0.0% | object |
| 2 | `attributes.title` | 20,000 | 0.0% | object |
| 3 | `attributes.translated_title` | 344 | 98.3% | object |
| 4 | `attributes.normalized_title` | 18,123 | 9.4% | object |
| 5 | `attributes.description` | 16,424 | 17.9% | object |
| 6 | `attributes.url` | 20,000 | 0.0% | object |
| 7 | `attributes.first_seen_at` | 20,000 | 0.0% | object |
| 8 | `attributes.last_seen_at` | 20,000 | 0.0% | object |
| 9 | `attributes.last_processed_at` | 20,000 | 0.0% | object |
| 10 | `attributes.contract_types` | 20,000 | 0.0% | object |
| 11 | `attributes.categories` | 20,000 | 0.0% | object |
| 12 | `attributes.onet_data.code` | 19,989 | 0.1% | object |
| 13 | `attributes.onet_data.family` | 19,989 | 0.1% | object |
| 14 | `attributes.onet_data.occupation_name` | 19,989 | 0.1% | object |
| 15 | `attributes.posted_at` | 8,788 | 56.1% | object |
| 16 | `attributes.recruiter_data.name` | 124 | 99.4% | object |
| 17 | `attributes.recruiter_data.title` | 120 | 99.4% | object |
| 18 | `attributes.recruiter_data.contact` | 124 | 99.4% | object |
| 19 | `attributes.salary` | 9,174 | 54.1% | object |
| 20 | `attributes.salary_data.salary_low` | 8,664 | 56.7% | float64 |
| 21 | `attributes.salary_data.salary_high` | 8,664 | 56.7% | float64 |
| 22 | `attributes.salary_data.salary_currency` | 8,664 | 56.7% | object |
| 23 | `attributes.salary_data.salary_low_usd` | 8,664 | 56.7% | float64 |
| 24 | `attributes.salary_data.salary_high_usd` | 8,664 | 56.7% | float64 |
| 25 | `attributes.salary_data.salary_time_unit` | 8,664 | 56.7% | object |
| 26 | `attributes.seniority` | 20,000 | 0.0% | object |
| 27 | `attributes.status` | 12,944 | 35.3% | object |
| 28 | `attributes.language` | 19,059 | 4.7% | object |
| 29 | `attributes.location` | 18,337 | 8.3% | object |
| 30 | `attributes.location_data` | 20,000 | 0.0% | object |
| 31 | `attributes.tags` | 20,000 | 0.0% | object |
| 32 | `relationships.company.data.id` | 20,000 | 0.0% | object |
| 33 | `relationships.company.data.type` | 20,000 | 0.0% | object |

#### `job_openings_included`: 10,907 rows × 5 columns (from the first 20,000 job-opening documents), duplicates removed by type and id

| # | Column | Non-null | Null % | Dtype |
|---|---|---|---|---|
| 0 | `id` | 10,907 | 0.0% | object |
| 1 | `type` | 10,907 | 0.0% | object |
| 2 | `attributes.domain` | 10,907 | 0.0% | object |
| 3 | `attributes.company_name` | 10,866 | 0.4% | object |
| 4 | `attributes.ticker` | 1,181 | 89.2% | object |
<!-- schema:end -->

---

## 3. Current status

**Done**
- [x] Environment set up, with S3 access working
- [x] Companies file loads fully and is cached as Parquet
- [x] Job openings sample loads (first 200,000 documents) and is cached as Parquet
- [x] File structure identified as JSON:API (`data` / `included`)
- [x] Flattened `data` and `included` into one row per record (companies: all documents; job openings: first 20,000 documents)
- [x] Generated the [Schemas](#schemas) section

**What the schemas already tell us** (from the first 20,000 job openings)
- `description` is 82% filled, so roughly 4 in 5 postings can be matched on full text.
- `onet_data` is 99.9% filled, so O*NET evaluation labels are viable.
- `seniority` is 100% filled, so we can filter the persona to entry-level roles.
- `posted_at` is only 44% filled, but `first_seen_at` is 100% filled. Use `first_seen_at` for "posted N days ago" on the card.
- `salary_data` is about 43% filled, so show salary on the card only where it's available.
- Every job opening links to a company through `relationships.company.data.id`. The company's name and domain come with the job opening itself.
- `contract_types` and `location_data` are 100% filled, so the persona can also be filtered by contract type and region.
- `url` is 100% filled, so every card can link to the original posting.
- `tags` and `categories` are 100% filled. If `tags` holds skills, it gives the gap analysis structured data.
- The sample is the first 20,000 documents, not a random sample. See [Data quirks](#data-quirks).

**Next** (the full task list is in [`PRIORITY_QUEUE.md`](PRIORITY_QUEUE.md))
- [ ] Team agrees on the [focused question](#1-goals) and the source background
- [ ] Finish the [feasibility checks](#week-1-corpus-profiles-evaluation-set)

---

## 4. Three-week plan

### Priority queue

The ordered list of every task, including repo setup and housekeeping, is in [`PRIORITY_QUEUE.md`](PRIORITY_QUEUE.md). It is the single source of truth for what to do next.

### Week 1: corpus, profiles, evaluation set

**Feasibility checks.** The EDA should answer these first:
- [x] How complete is `description`? 82% filled. Still to check: how long descriptions are.
- [ ] What share of postings are in English? Use `attributes.language`.
- [x] How complete is `onet_data`? 99.9% filled. Still to check: how many O*NET families appear.
- [ ] How many postings fall in plausible target families for the chosen persona?
- [x] How do job openings link to companies? Through `relationships.company.data.id` to the company `id`.
- [ ] What share of job-opening companies are in the companies table?
- [ ] Is the sample representative? Check how postings are spread across companies and `first_seen_at` dates. If the file is sorted, switch to random sampling while streaming.
- [ ] What do `status` values mean, and how many postings are still live based on `last_seen_at`?
- [ ] What do `contract_types` and `location_data` contain, and which values define the persona?
- [ ] What do `tags` and `categories` contain? Are they skills, labels, or derived from O*NET?

**Build:**
- [ ] Choose the source background (see the [Decision log](#8-decision-log))
- [ ] Corpus: live, English postings with real descriptions, filtered by the persona's seniority, contract type and region. Stream the full file once with filters if the sample is too small or unrepresentative.
- [ ] Profiles: 10–20 in total, mixing de-identified team résumés with synthetic profiles covering different backgrounds
- [ ] Text cleaning and duplicate removal (see [Text preprocessing](#text-preprocessing))
- [ ] Each profile's obvious O*NET families, assigned before any run
- [ ] Rating rubric written and agreed (see [Evaluation](#5-evaluation))
- [ ] Move shared loaders (`fetch`, `stream_jsonl`, `load_cached`, `unpack`) into `src/`

**Exit criteria:** corpus built, profiles written, evaluation plan agreed.

### Week 2: matching

- [ ] Baselines: random, title keyword search, TF-IDF + cosine
- [ ] Challenger: text embeddings, run locally (see [Embedding configuration](#embedding-configuration))
- [ ] Hybrid: reciprocal rank fusion of TF-IDF and embeddings
- [ ] Blind, graded team ratings of the pooled top 10
- [ ] Score all methods on the [evaluation metrics](#5-evaluation)
- [ ] **Core finding:** relevant roles that embeddings find and TF-IDF misses, with examples and an explanation of why

**Exit criteria:** comparison results written up.

### Week 3: explanations, card, pitch

- [ ] Extract requirements from each description and label them **required** or **preferred**, using section-header rules first (see [Explanations and gaps](#explanations-and-gaps))
- [ ] Flag "N+ years" experience requirements
- [ ] Gap analysis: compare profile against requirements
- [ ] Match explanations, each grounded in quoted profile and job text, with an automatic grounding check
- [ ] Assemble the [recommendation card](#the-recommendation-card) and the demo
- [ ] Final pitch: each team member pitched for a role the system recommended

**Exit criteria:** demo, write-up and pitch ready.

---

## 5. Evaluation

Without labeled relevance, "embeddings are better" is only an opinion. We use two kinds of labels:

1. **O*NET labels (automatic).** Before any run, each profile is assigned its **obvious O*NET family or families**: what the person would search for today. A recommendation counts as a **discovery** when its `onet_data.family` falls outside those families. This measures whether a method finds roles beyond the expected job titles.
2. **Team relevance ratings (manual).**
   - **Pool:** the top 10 from every method for each profile, combined and shuffled, with the method hidden.
   - **Graded scale:** 0 = not relevant, 1 = stretch (plausible with clear gaps), 2 = strong match.
   - **Rubric:** written and agreed **before** rating. It defines each grade, with one worked example.
   - **Agreement:** two people rate the same ~20% of items. Report Cohen's kappa. If it's low, refine the rubric and rate again.

**k = 10 throughout,** matching what the card shows.

**Metrics:**
- **Precision@10:** the share of the top 10 rated 1 or 2
- **nDCG@10:** rewards strong matches ranked higher, using the graded ratings
- **Discovery rate@10:** the share of the top 10 rated 1 or 2 **and** outside the obvious families
- **Family diversity@10:** the number of distinct O*NET families in the top 10
- **Unique finds:** relevant postings found by only one method

### Hypothesis

Write this down, and agree on it, **before** running any model:

> Embeddings achieve a higher **discovery rate@10** than both TF-IDF and title keyword search on the test profiles, and the 95% paired bootstrap confidence interval for the difference excludes 0. Precision@10 does not drop meaningfully: more discovery shouldn't come from more irrelevant results.

If the result is negative, that's still a finding. Report it, then use the error analysis to explain why.

### Baselines

A method only counts as better if it beats something simpler.

| Method | Role |
|---|---|
| Random ranking | Sanity floor; every method must clear it |
| **Title keyword search** | How a career changer searches today. Beating it supports the "roles you didn't know to search for" claim. |
| TF-IDF + cosine | Standard text-matching baseline |
| Embeddings | Challenger |
| Hybrid: reciprocal rank fusion of TF-IDF and embeddings | Combines the exact-skill matching of keywords with the meaning-based matching of embeddings. Cheap, and often the strongest. |

### Text preprocessing

The same preprocessing applies to every method.

- **Clean postings:** strip HTML and collapse whitespace. Remove recurring boilerplate (equal-opportunity statements, benefits sections) where it can be detected by section headers.
- **Remove duplicate postings:** the same job often appears more than once (reposts, multiple locations). Treat postings with the same company, the same normalized title text and near-identical descriptions as one. Otherwise duplicates fill the top 10.
- **Build the profile text consistently:** the same structure for every profile, for example summary + experience bullets + skills. Synthetic and real profiles use the same template.

### Embedding configuration

- **Model:** a small local `sentence-transformers` model. Start with `all-MiniLM-L6-v2` (fast) and compare one stronger small retrieval model, such as `bge-small-en-v1.5` or `e5-small-v2`, on dev profiles. Pin the winner.
- **Long descriptions:** these models only read the first ~256–512 tokens. Split descriptions into chunks, embed each chunk, and score a posting by its best-matching chunk. Compare this with plain truncation on dev profiles.
- **Prefixes:** models like e5 and bge expect "query:" / "passage:"-style prefixes. Use them as each model's documentation specifies.
- **Normalize** vectors, so a dot product equals cosine similarity.
- **Cache** embeddings in `data/`, keyed by posting ID, so they're computed once.

### TF-IDF configuration

A weak TF-IDF baseline would make embeddings look better than they are. Start from these defaults (`TfidfVectorizer`) and tune on dev profiles only:

| Setting | Start | Why |
|---|---|---|
| `ngram_range` | `(1, 2)` | Captures phrases like "customer success" and "project management" |
| `sublinear_tf` | `True` | Damps repeated keywords in postings and résumés |
| `max_df` | about `0.5` | Drops boilerplate that appears in many postings, such as equal-opportunity statements, benefits text and "apply now" wording |
| `min_df` | `2`–`5` | Drops typos and one-off terms |
| `stop_words` | `"english"` | Standard |

**Rules**
- **Fit the vectorizer on the postings only,** then transform the profiles with it. Fitting on postings and profiles together lets the profiles change the term weights.
- **Tune both methods equally hard.** Embeddings have settings too: which text fields to embed and how long descriptions are cut off.


### Profile split

Nothing is trained on labels, so there's no train/test split of postings. Profiles are the unit of evaluation instead:

- **Dev profiles:** about a third of them. Use these to tune every choice: preprocessing, k, the embedding model, which text fields to use.
- **Test profiles:** the rest. Run them **once**, at the end, with the settings fixed.

### Statistics

- Compute each metric per profile, then report **mean and 95% paired bootstrap confidence interval** across test profiles.
- Use a **paired permutation test** for "is method A better than method B?"
- Keep comparisons few: embeddings vs. each other method (random, title search, TF-IDF, hybrid). Correct across them with Benjamini–Hochberg.
- With 10–20 profiles, confidence intervals will be wide. State that openly rather than overclaiming.

### Ablations

Pick 2–3. Run them on dev profiles only.

- TF-IDF with and without bigrams
- Embedding the title only, the description only, or both
- With and without the requirements section of the description
- **LSA:** TF-IDF reduced with `TruncatedSVD`, a middle ground between keywords and embeddings. It shows whether embeddings win because of meaning or because of vocabulary. Use `TruncatedSVD`, not PCA, on sparse TF-IDF, and choose the number of components on dev profiles.
- **BM25 vs. TF-IDF cosine:** BM25 is the standard search-engine keyword ranker. It checks that the keyword baseline isn't weak (`rank-bm25` package).


### Error analysis

The core finding. Done on test results after the statistics:

- **Unique finds:** for each method, the relevant postings only it found. Explain why the other method missed them.
- **Failures:** the bad results in each method's top 10.
- **Failure types, with counts:**
  - Vocabulary mismatch (same skill, different words)
  - Boilerplate dominating the match
  - Seniority or experience mismatch
  - Matches on a title word only
  - Duplicate postings

Counts turn a list of anecdotes into a finding.

### Explanations and gaps

- **Required vs. preferred, rules first:** split descriptions on section headers such as "Requirements", "Qualifications", "Preferred", "Nice to have" and "Bonus". Measure how often this works on dev postings before considering GenAI.
- **Experience flag:** pull "N+ years" requirements with a regular expression and show them on the card. They're the most common blocker for career changers.
- **Grounding check:** every phrase an explanation quotes must appear word for word in the profile or the posting. Check this automatically, and report the pass rate.

### Leakage rules

- **Never put `onet_data.*` into the matching text.** It's the evaluation label.
- Check whether `normalized_title`, `categories` and `tags` come from O*NET. Keep any that do out of the matching text too.
- Never tune on test profiles.
- Raters never see which method produced a result.

### Reproducibility

- Package versions are pinned in `requirements.txt`.
- Pin the embedding model's name and version.
- Save the list of posting IDs in the corpus, so every method ranks the same set. That includes title keyword search: it ranks the same description-filtered postings, even though it only reads titles.
- Fix random seeds, and record each run's settings with its results.

---

## 6. Roles

| Person | Focus | Owns |
|---|---|---|
| Jacob | EDA, data pipeline, feasibility checks | `jacob_eda/`, loaders in `src/` |
| TBD | Matching: TF-IDF and embeddings | |
| TBD | Evaluation: profiles, labels, metrics | |
| TBD | Explanations, card, demo | |

---

## 7. Scope

- One source background
- Role discovery for that background
- TF-IDF vs. embeddings comparison
- Grounded explanations and gap analysis
- Recommendation card with the company and posting details we actually have

**How to describe the machine learning:** unsupervised representation learning for retrieval (TF-IDF, LSA, embeddings), evaluated against held-out labels. O*NET labels and team ratings are used only for evaluation, never for training.

---

## 8. Decision log

| Date | Decision | Why | Status |
|---|---|---|---|
| 2026-10-05 | Cache parsed datasets as Parquet in `data/` | Parsing the JSON takes about 10 minutes per run | Adopted |
| 2026-10-05 | Use a 200,000-document sample of job openings for EDA | The full file is about 19 GB compressed | Adopted |
| 2026-10-06 | Remove duplicate `included` objects before flattening | Flattening every copy first crashed the kernel | Adopted |
| 2026-10-06 | Narrow focus with maximum depth: one persona × one search decision | Three weeks isn't enough for breadth | Adopted |
| 2026-10-06 | Focus: career-transition role discovery, TF-IDF vs. embeddings, grounded explanations | Merges both plans; the depth comes from the measured comparison | Proposed |
| 2026-10-06 | Use `first_seen_at` for posting age, not `posted_at` | `posted_at` is only 44% filled; `first_seen_at` is 100% | Proposed |
| 2026-10-06 | Recommend only live postings (`status` / `last_seen_at`) | A closed posting is a useless recommendation | Proposed |
| 2026-10-06 | k = 10; graded relevance (0/1/2) with a written rubric and a rater-agreement check | Matches the card; graded ratings allow nDCG; agreement makes manual labels credible | Proposed |
| 2026-10-06 | Add a hybrid method (reciprocal rank fusion of TF-IDF and embeddings) | Cheap third contender that is often the strongest | Proposed |
| 2026-10-06 | Adopt the [evaluation protocol](#5-evaluation): written hypothesis, title-search baseline, dev/test profile split, paired bootstrap, leakage rules | Makes "embeddings are better" a measured claim rather than an opinion | Proposed |
| | Source background for the persona | | TBD |
| | Embedding model, which must run locally | Compliance | TBD |
| | Tool for processing the full job openings file | | TBD |
| | Virtual environment name: README says `.venv`, Jacob uses `PL` | Consistent setup across the team | TBD |

---

## 9. Risks and compliance

| Risk | Mitigation |
|---|---|
| **Sending real job data to outside AI services** | The advisors' rules (in `CLAUDE.md`) allow AI tools for coding, but no confidential data may be uploaded to them. Run embeddings locally, for example with `sentence-transformers`. Show GenAI explanations on synthetic postings only. |
| **AI coding tools on project files** | Allowed for coding, brainstorming and debugging, per `CLAUDE.md`. Claude never reads `data/` or `.env`, and works from code, column names, dtypes and shapes only. |
| Personal details in résumés | Remove names, contact details, employers, schools and locations before use. Store résumés like data: never commit them. |
| Biased evaluation | Hide which method produced each result while rating. Use O*NET labels as an independent check. |
| Biased sample: the first N documents, not random | Check company and date spread. If the file is sorted, sample randomly while streaming the full file. |
| Too few postings for the persona | The Week 1 feasibility checks catch this. Fall back to streaming the full file with filters, or choose another background. |
| Data leaves approved locations | Keep data only in `data/` or S3. Never commit data. Clear notebook outputs before every commit. |
| Notebook kernel crashes lose in-memory work | Parquet cache makes reloading fast. Run cells from the top after a restart. |
| End-of-project data deletion | Track every local copy (`data/`, downloads, résumés) so all of them can be deleted and confirmed in writing |

---

## 10. Open questions

**For the team**
- Do we agree on this focus?
- Which source background do we choose?
- Who owns which workstream?
- What meeting cadence should we use, and where do we discuss async?
- What format should the demo take: a notebook, or a small Streamlit app?

**For PredictLeads**
- Does role discovery for career changers fit what they want from the project?
- Is schema documentation available for both datasets, especially `onet_data`?
- Are daily snapshots full copies or incremental changes?
- What do `status` values mean, and how are `tags` and `categories` produced?
- Is the job openings file sorted in some way, for example by company or date?

---

## 11. Discussion notes

Add the newest meeting at the top. Copy this template:

```markdown
### YYYY-MM-DD: <meeting topic>
**Attendees:**
**Discussed:**
-
**Decisions:** (also add these to the Decision log)
-
**Action items:**
- [ ] <task> (owner, due date)
```
