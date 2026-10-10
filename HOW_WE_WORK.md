# How We Work

PredictLeads × TripleTen externship · **How we work.** Last updated 2026-10-09.

> A living document. The career-changer focus is part of the project description; the rest of the current choices are Jacob's working proposal.
> Follow the [Data & Security Rules](README.md#data--security-rules): no data values, `df.head()` output or screenshots in any doc.

## Where everything lives

| File | Its one job |
|:--|:--|
| [`PROJECT_OUTLINE.md`](PROJECT_OUTLINE.md) | **What we're doing:** the idea, findings so far, factors, the notebook series. Start here |
| `HOW_WE_WORK.md` (this file) | **How we work:** standards, scope, roles, risks |
| [`docs/decisions.md`](docs/decisions.md) | Every decision, with its reasoning |
| [`docs/data.md`](docs/data.md) | Data sources, fields, quirks and auto-generated schemas |
| [`DIFFERENTIATORS.md`](DIFFERENTIATORS.md) | **How we stand out:** the five angles, the pitch, how we talk about the data, and what we never claim |
| [`docs/fit_analysis.md`](docs/fit_analysis.md) | Transferable fit: Option A vs. B, and the applicant profiles (real career switches from a public resume dataset; volunteer profiles later) |
| [`PRIORITY_QUEUE.md`](PRIORITY_QUEUE.md) | What to do next |
| [`APPLICATION.md`](APPLICATION.md) | The web app's design |
| [`README.md`](README.md) | Setup and the data rules |
| `notebooks/` | EDA_SDA_1–4, TimeSeries, ML: the analysis itself. Each has its own detailed plan. `Profiles.ipynb` finds real career switches in a public resume dataset, for the fit model |

## Goal and success criteria

For a given **career changer** (anyone switching from their current background to a different role or field): suggest the computer and mathematical roles their skills transfer to, and rank listings in those roles by which to apply to first, and why (details in the outline). We succeed if:
- The prioritization model ranks listings better than how a career changer searches today (newest first, title keyword search) on a later time period, using outcomes we can measure (such as whether a listing is still open).
- Every ranked listing shows how open it is to a career changer, from text features validated in EDA_SDA_3.
- Listings a career changer can't get are hidden, using only reliable or validated rules, with a count of what each rule hides.
- Role discovery ranks the role real career changers actually moved into near the top (hit rate in the top 3 and top 5), better than a title-keyword baseline.
- Every factor in the model is backed by a measured pattern with a confidence interval, or reported plainly as having no usable signal.
- Findings are nuanced: factors matter differently by role, source or listing type.
- Every ranking can be explained: each listing's score shows the factors behind it.
- Claims stay within what the data measures: listings appearing and disappearing, not hires.

## Working standards

- **Course baseline:** the TripleTen course templates set the structure and methods for every notebook (intro, clarifying questions, a comment after every chart, documented decisions, baselines, train/validation/test used once, a conclusion). Every applicable course method is used, then taken further.
- **Reasoning for every decision,** recorded where the decision is made and in `docs/decisions.md`.
- **Fix data quality as we go; apply scope filters at the end.** Cleaning fixes broken values. Scope decisions are defended first, then applied all at once in a final-filters section.
- **Hide, never remove.** Listings that are stale (more than 90 days old on the date viewed), likely ghosts, or out of a user's reach are hidden in the app, with a count and a reason, and stay in the analysis and the model. Never drop listings by how long they ended up open: that uses the future and biases the timing model.
- **Each analysis states its view of the table** (for example, originals only for counts; listings with `posted` for day of week) to avoid skew.
- **Each notebook hands a final table to the next,** saved under `data/clean/`.
- **Shared code lives in `src/`** (`eda_helpers.py`, `pl_data.py`, `career_switch.py`, `export_markdown.py`), not copied between notebooks.

## Validation design

- **Temporal holdout:** train on earlier months, test on later ones. A ranking has to work going forward.
- **Folds grouped by company,** so models learn about postings, not one company's habits.
- **Censoring:** still-open postings are kept and handled by survival methods, never dropped.
- **Baselines:** the segment's overall average, last year's value for seasonality, newest-first and title keyword search for prioritization.
- **Statistics:** bootstrap 95% confidence intervals; Benjamini–Hochberg correction when testing many features or segments.
- **Confounders: compare within groups, don't filter.** Every factor shows its raw gap, then the gap within company-volume band × source type (and role bucket where possible); it counts only if the gap holds. Standard confounders: company volume, source and platform, staffing agencies, role bucket and seniority, month first seen (list in EDA_SDA_2 section 2). EDA_SDA_4 confirms the survivors in one Cox survival regression adjusting for all of them, with company-clustered errors.

### Leakage rules

- **Features may only use what's known when a listing first appears:** its text (including the career-change flags), seniority, source, company, pay, `first_seen_at`, and repost history from *earlier* copies. Never use `last_seen_at`, `status`, days open, or whether a listing is reposted *later* as a feature for predicting timing.
- **Counts over time use only earlier listings:** company hiring volume and reposts count only listings first seen before the listing (or before the chosen date). Whole-window counts are descriptive only. EDA_SDA_1 Q1d rechecked its company-volume finding this way: 13 vs. 30 days (20+ openings vs. one-off), so it holds.
- **Date picker (as-of date):** a listing is shown on a date if it was first seen on or before it and last seen on or after it. Its features are computed as of that date; `last_seen_at` only decides which listings were live and never feeds a score. The model that scores a date is trained only on what was known by then: listings still open that day are censored at that day. Dates before the test period need a model retrained as of that date.
- **Career-switch counts during evaluation:** compass and path counts used as features are built from the training people only, so a test person never counts toward their own answer.
- **Job growth expectations** are published projections, the same for every listing in a role; they come from outside the listing data, so they don't leak listing outcomes.
- **Never tune on the test period**, and use the test set once, at the end.
- **Applicant profiles:** each synthetic profile stays in the same fold as the real profile it came from, and final results are reported on real profiles only.

## Reproducibility

- Package versions are pinned in `requirements.txt`.
- Save the list of posting IDs in the analysis table, and the snapshot date.
- Fix random seeds, and record each run's settings with its results.

## Scope

- **Applicant side only.** "Best time to post" for employers would need hiring outcomes we don't have.
- **English-language listings in the United States, first seen April–September 2026,** plus the data's first two days (2026-03-30 and 31), kept and flagged `partial_month` (decisions and reasons in `docs/decisions.md`).
- **Applicants are career changers:** anyone switching from their current background to a different role or field. Source backgrounds are the team's own. Listings a career changer can't get are hidden; the model weighs everything else.
- **Target: computer and mathematical (US).** EDA_SDA_2 splits it into role buckets shared with the career-switch profiles (not one role, since role discovery needs several), keeping buckets with enough data on both sides once unreachable listings are hidden.
- **Transferable fit and role discovery** learn from real career switches in a public resume dataset (`notebooks/Profiles.ipynb`: 2,813 career changers: 764 who retrained from another field and 2,049 who moved between tech roles, sideways or up; rules spot-checked). Volunteer profiles, collected with consent once the advisor approves, are a recent test set. Approach: [`docs/fit_analysis.md`](docs/fit_analysis.md).

**How to describe the machine learning:** survival and time series analysis of listing activity, plus supervised learning (classification, survival and learning-to-rank models) on listing features and text, including career-change accessibility, validated on later months.

## Roles

| Person | Focus | Owns |
|---|---|---|
| Jacob | Data pipeline, EDA/SDA | `notebooks/`, `src/` |
| TBD | Time series | `notebooks/TimeSeries.ipynb` |
| TBD | Text features, fit and ranking | `notebooks/EDA_SDA_3.ipynb`, `notebooks/ML_Fit.ipynb`, `notebooks/ML_Rank.ipynb` |
| TBD | Web app and pitch | `APPLICATION.md` |

## Risks and compliance

| Risk | Mitigation |
|---|---|
| **Sending real job data to outside AI services** | The advisors' rules (in `CLAUDE.md`) allow AI tools for coding, but no confidential data may be uploaded to them. Run any embeddings locally. |
| **AI coding tools on project files** | Allowed for coding, brainstorming and debugging, per `CLAUDE.md`. Claude never reads `data/` or `.env`, and works from code, column names, dtypes and shapes only. |
| **Crawler timing mistaken for employer behavior** | Day-of-week patterns use `posted_at` (read as Central European dates), never `first_seen_at`. |
| **Disappearing mistaken for filled** | Word every claim as "postings close" or "disappear", never "get filled". State it in the limitations. |
| **Too little history for seasonality** | Only six months exist, so no seasonal claims unless PredictLeads provides older snapshots. |
| **Look-ahead leakage in timing models** | Follow the leakage rules above: features only from what's known when a posting appears. |
| Biased sample: the first N documents, not random | EDA_SDA_1 found no sort order by date or company. Recheck if the sample changes. |
| Too few postings in the chosen segment | EDA_SDA_2's depth check sizes it. Fall back to streaming the full file with filters, or a bigger segment. |
| Data leaves approved locations | Keep data only in `data/` or S3. Never commit data. Clear notebook outputs before every commit. |
| Notebook kernel crashes lose work | Flattening resumes from saved parts; cleaning is cached. Run cells from the top after a restart. |
| Career-change flags are keyword rules with false positives ("bachelor's degree or equivalent experience required" counts as both) | EDA_SDA_3 validates them: a team member checks a random sample of matches locally |
| Accessibility read from what listings ask, not from who gets hired | Word claims as "open to newcomers on paper", never "hires career changers". State it in the limitations |
| Hiding listings on keyword rules hides some a career changer could get | Hide only on reliable (seniority) or validated (EDA_SDA_3) rules; report how many listings each rule hides |
| Applicant profiles are personal data | Collect only with written consent and the advisor's approval; keep them under `data/`; never commit them or share them with AI tools; use synthetic profiles in anything presented |
| End-of-project data deletion | Track every local copy (`data/`, downloads) so all of them can be deleted and confirmed in writing |

## Discussion notes

Add the newest meeting at the top. Copy this template:

```markdown
### YYYY-MM-DD: <meeting topic>
**Attendees:**
**Discussed:**
-
**Decisions:** (also add these to `docs/decisions.md`)
-
**Action items:**
- [ ] <task> (owner, due date)
```
