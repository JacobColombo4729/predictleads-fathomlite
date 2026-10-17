# PredictLeads Career Intelligence Roadmap

Updated: October 10, 2026  
Status: Working roadmap, rewritten to match the app as built. What's left to do is in the app's priority queue, [APP.md](APP.md). For what each ML piece is and how well it is validated, see [docs/ml-inventory.md](docs/ml-inventory.md).

**Audience:** career changers, starting with ourselves (TripleTen AI/ML graduates), looking for realistic roles in AI/ML, data, and technical sales / solutions engineering. The same hiring data also serves PredictLeads' own customers (see Signals).  
**Target date:** about 2026-10-23. Deployment target and demo format are not yet decided.

## Product Goal and User Journey

Help a career changer find relevant listings, understand the evidence, see which companies are actively hiring, and know what to do next.

**Profile → Discover (ranked jobs) → job details → Saved → Compass (analysis of picked jobs)**, with **Signals** (who is hiring, and how) alongside.

1. **Profile** (built): blank start; fill the form or use **Magic fill** on a PDF/text resume; entries are typed Job / Project / Education; desired roles; Save; **Start over**.
2. **Discover** (built): hybrid ranker over about 81,000 historical listings; title-first search; filters; 👍/👎 relevance feedback adjusts the list live; a company hiring badge on each card.
3. **Job details** (built, basic): description, source link, historical-snapshot labelling, a keyword skills check, and a **Company tab with the company's hiring activity** (pattern, weekly sparkline, example openings, technologies named).
4. **Saved jobs** (built): stages and notes; hiring badges; tick up to 20 and send to the Compass.
5. **Compass** (built, local, no LLM): themes across the picks, a compass map, fit tiers, what the picks ask for next to the person's own evidence, next steps and "hidden fit" suggestions.
6. **Signals** (built): companies ranked by hiring bursts and steady hiring in AI/ML, data and technical sales, with **Ask the market** on top (2026-10-10): ask in plain words, by typing or by pressing Talk, answered only from our data, with an optional sourced "Research this company" step ([docs/ask-the-market.md](docs/ask-the-market.md)).
7. Dark theme by default, with a toggle.

Status terms: **Built** means working in the running app and verified; **Implemented** means code exists but is not integrated or evaluated; **Experimental** means bounded evidence, not validated quality; **Planned** means no implementation yet.

## 1. Data and Candidate Profiles

**Built**
- Full snapshot scan: 13,415,388 job listings read to end of file with no errors (snapshot 2026-09-30). A targeted pool for AI/ML, data and technical-sales roles has 319,970 matches (`code/extract_targeted_pool.py`).
- Ranking corpus: 244,558 unique listings (targeted pool plus a random 30,000 background sample), usable descriptions only, exact duplicates collapsed. 163,274 listings are explicitly `closed` and excluded; the **81,284 listings with unknown status** form the serving set.
- Candidates: 10 distinct candidates = 12 profiles (2 real and consenting, Jesse and Jacob; 8 fictional). Real resumes and profile text stay in the Git-ignored `data/` folder.
- Time fields: `first_seen_at` / `last_seen_at` are crawler observation times, not posting or closing dates ([docs/signal-semantics.md](docs/signal-semantics.md)). Openness is never inferred from dates or null status. They now feed the Signals layer.
- Data sharing: `data/` and `pretrained/` are git-ignored (NDA data). Teammates receive them as zip files over a private channel; the NDA forbids only public posting.

**Open**
- Companies are only the same-wrapper company reference on each listing; the separate companies-file join is not validated.
- Duplicates are collapsed only when company, title and description are identical; reposts with edited text remain and can still inflate "new openings" in Signals.

## 2. ML Matching

**Built: the demo ranker**
- Equal-weight reciprocal-rank fusion (depth 50, k=60) of TF-IDF cosine and BGE (`bge-base-en-v1.5`) cosine. Scores are retrieval signals, not match percentages or hiring probabilities.
- Relevance feedback (Rocchio-style, liked 1.0 / disliked 0.5) adapts one session; nothing is retrained.
- Baselines: keyword overlap and TF-IDF cosine in `code/job_matching_engine.py`; the backend can serve TF-IDF alone with `.\run_demo_backend.ps1 -Ranker tfidf`.

**Evaluated against real outcomes, not labels** (`analysis/PROJECT_OUTLINE.md`, How we keep it honest). For 2,813 real career changers we know the role each moved into; on held-out people, ML_Fit's role-level TF-IDF + BGE fusion puts it in the top 3 for 70.0% and the trained layer for 81.7%, against 63.2% for the most-common-role baseline. The app's rankers are now scored on the same career moves (`analysis/notebooks/Evaluate.ipynb`): the text match alone 66.3% at top 3 (not shown above the baseline), ML_Rank's queue on top 69.8%, the combined model 76.4%. The demo and write-up say so. The blinded review page is an optional usability check, not the evaluation.

**Built: role-family classifier (supporting model, not in the app)**
- Logistic regression predicts 17 classes (16 role families plus Other) from the job title, trained on the provider's O*NET label. Leak-free test accuracy about 72% (TF-IDF on title; BGE title embeddings 70.5%) vs 30% for the majority class, company-held-out, 3 splits. **The earlier 94.7% was inflated:** its text included the occupation name the labels come from (corrected 2026-10-13). Measures agreement with provider labels, not fit.

**History, not evidence.** Earlier experiments (MiniLM, section-aware text, cross-encoder, weighted RRF, faceted text) were small and inconclusive; see [docs/archive/journal.md](docs/archive/journal.md). Do not quote them.

## 3. Signals layer (hiring momentum) — Built and backtested

- `code/hiring_signals.py` (offline), `backend/signals.py`, `/api/v1/signals/*`, `frontend/src/pages/SignalsPage.tsx`. Report: [docs/hiring-signals.md](docs/hiring-signals.md).
- Per company: new openings in 30 days against its own prior pace, labelled burst / sustained / cooling / quiet; reposts merged; bulk-import days and crawler dropouts excluded; no look-ahead.
- **Backtest result (honest):** no model beats "openings last month" for predicting volume. A burst is a timing trigger, not a forecast: bursts keep about 64% of their volume the next month; sustained hirers keep about 99%.
- **Rule: nothing inferred.** The tab shows only counts, dates, example listings and technology tags from the data. Inferred statements such as what a company is "likely buying" were removed (2026-10-10).
- Limits: four months of data, one test month, crawler dates.

## 4. GenAI Features

**Feature 1: resume "Magic fill" — Built and live-tested.** Text PDF/`.txt`/`.md`; contact details removed; `gpt-4.1-mini`, temperature 0, strict JSON; per-use consent; every item needs a verbatim quote or it is dropped. Off unless started with `-ResumeFill`. Budget: original $3 used; extra $2 approved 2026-10-09, a few cents spent.

**Compass — Built (local, no LLM).** Clustering and similarity over stored BGE vectors. Thresholds are heuristics, not validated. The **stretch slider** (added 2026-10-10) asks the backend for suggested roles further from the picks when released; the tier board shows picks and suggestions with fixed cutoffs (Best fit 75+, Reach 55-74, Stretch below 55). The map's spokes are role families with a turnable scope ring (Jacob, `feature/mac-support`, merged 2026-10-10). Each role shows the company hiring badge. **Add to my tiers** (2026-10-10): a suggested role can be added to Your fit, in tiers; it leaves the "Roles you didn't pick" list, shows as an Added card (with Remove), survives moving the slider, and is included in the AI skills gap analysis.

**Feature 2: AI skills gap analysis in the Compass — Built, awaiting a live test (2026-10-10).** One OpenAI request covers every job sent to the Compass (up to 20) and the result appears in a **Skills gap analysis for each job** section on the Compass page, grouped per job: each requirement is labelled supported / not shown in your profile / conflicting, shows the exact profile line and listing line it relied on, and each job lists carry-over experience and next steps. Code: `explain_matches` in `code/grounded_match_explanations.py`, `POST /api/v1/profiles/{id}/compass/gap-analysis`, `frontend/src/components/compass/GapSection.tsx`. Works for whoever's profile is in the session. Off unless the backend starts with `.
un_demo_backend.ps1 -Assess`; per-use consent; sends the profile's experience and skills (no name, email, phone or links) and the job descriptions (trimmed to 8,000 characters each) to `gpt-4.1-mini`; every cited line must exist or that job's result is dropped (the other jobs still show); a result with any dropped job is not cached so running again retries it; results cached per session. Tested offline with a mock provider and in the browser with a canned response; **not yet run live** (a 20-job run may take a minute or two; if too slow, split into parallel batches). The Compass itself uses no LLM, so this and Magic fill are the two GenAI features. The older single-job endpoint remains but has no UI.
un_demo_backend.ps1 -Assess`; per-use consent checkbox; sends the profile's experience and skills (no name, email, phone or links) and the job description to OpenAI (`gpt-4.1-mini`); every point must cite lines that exist in the profile or listing or the whole result is rejected; results are cached per session. Tested offline with a mock provider and in the browser with a canned response; **not yet run live**. The Compass uses no LLM, so this and Magic fill are the two GenAI features.

## 5. Backend and Frontend

**Built**
- FastAPI backend: anonymous cookie sessions in SQLite, profiles, search, recommendations (RRF or TF-IDF), feedback, saved jobs, Compass, signals, assessment endpoint (provider off by default), resume extraction. All offline tests pass (`.\run_tests.ps1 -Frontend`).
- React + Vite frontend: Discover, Saved, Compass, Signals, Profile.
- Run locally: `.\run_demo_backend.ps1 [-ResumeFill]` (a few minutes to load, about 2.5 GB memory) and `cd frontend; npm run dev`, then open http://127.0.0.1:5173. If `localhost` misbehaves, use `127.0.0.1` and make sure no old dev server holds port 5173.

**Known gaps**
- The frontend README needs a Signals section; sessions are anonymous and temporary; the frontend has no automated tests (type-check and manual use only).
- Signals "burst" can fire on modest jumps at very large companies (e.g. 330 to 389 a month); a minimum-size rule would tighten it.

## 6. Remaining Work, in Order

Moved to the app's priority queue, [APP.md](APP.md).

## 7. Ideas for Later, Ranked by Value

Ranked on: value to the demo and to PredictLeads, effort, whether we can check it against data, and fit with our rule of **no inferred claims** (anything shown should be a count, a quote, or a clearly labelled model measurement).

| # | Idea | Value | Effort | Can we verify it? | Fits "no inference"? |
|---|---|---|---|---|---|
| 1 | **Near-duplicate / repost detection** (MinHash or embedding similarity on listing text) | High for PredictLeads: raises the accuracy of every count on Signals; data-quality story | About 1 day | Yes: sample matches by hand, and see how counts change | Yes: it removes inflation, adds no claims |
| 2 | **Skills across top jobs** (run the Compass requirement analysis over a user's top Discover results, show the most frequent requirements and the verbatim profile evidence) | High for users: the core "what should I learn" answer; reuses existing Compass code | About half a day | Partly: same heuristics as the Compass | Yes: shows frequencies and the person's own words |
| 3 | **Listing age and lifetime** (first descriptive: "observed for N days"; later a survival model for how long listings stay up) | Medium-high for PredictLeads; separates fresh from stale listings | Descriptive: a few hours. Model: 1-2 days | Yes: temporal holdout | Descriptive yes. A model prediction is inference: show it only as a labelled estimate, if at all |
| 4 | **Hiring-pattern clustering** (discover burst, steady, seasonal shapes from weekly curves) | Medium; improves Signals' credibility over hand-set thresholds | About 1 day | Yes: compare with the current rules | Yes: descriptive |
| 5 | **"Refine my matches"** (per-user model from 👍/👎; plan in the section below) | Medium-high for demo impact; evidence is thin without ratings | About half a day | Only with an opt-in rating export | Mostly: it is personalised ranking, labelled as such |
| 6 | **Profile what-if** (how would adding one skill change the ranking) | Low-medium | 1 day | Hard | Borderline: speculative |
| 7 | **Salary estimator** (fill missing pay with a regression) | Medium for users | 1-2 days | Yes: error ranges | **No**: it fills a blank with an estimate. Only acceptable if clearly labelled "estimate"; we recommend skipping |
| 8 | **Grounded resume bullets** (GenAI) | Medium | 1 day + consent and budget | Hard | Risky: generated text, needs strict grounding |

**Suggested order if time allows:** the ranker measurement (section 6, item 1), then ideas 2 and 1, then 3 (descriptive part), then 5. Skip 7 and 6 for the demo.

**Per-user preference model ("Refine my matches") — design notes, decision pending (added 2026-10-09)**
- After about 20-30 👍/👎 ratings, a button trains a small model on that user's ratings and re-ranks their results. Today's feedback is only a live Rocchio nudge.
- Design: unlock at about 10 ratings with at least 5 of each kind; heavily regularized logistic regression or ridge on BGE vectors of rated jobs; cross-validate on the user's ratings and refuse to apply the model if it does not beat chance; fuse with the RRF rank; show "refined with N 👍 and M 👎, predicted your ratings with X% accuracy, K of your top 10 changed"; keep some variety.
- Prerequisite: the rating log, **built 2026-10-10** ([docs/feedback-data.md](docs/feedback-data.md)); it records what was shown, ranks, ratings (with a reason for each thumbs-down), saves and Compass runs, and Discover mixes in "Something different" listings where the keyword and meaning signals disagree, so ratings can compare the methods. Production needs users to opt in (the switch exists). Risks: small samples, selection bias, ambiguous 👎, no accounts, consent for storing a taste model.

**Not planned**
- Career Adjacency graph (needs validated duties and O*NET assignments); market momentum across snapshots (only one snapshot); PDF OCR; accounts and persistent profiles; deployment hardening.

## Preserved Decisions and Limitations

- Keep missing, null, blank, invalid and unknown distinct. Do not infer that a listing is open from null status or from dates; every listing is a historical record from the 2026-09-30 snapshot.
- **Do not show anything that is inferred without evidence** (decided 2026-10-10): no "likely buying", no guessed intent, no filling blanks with estimates unless clearly labelled as a model estimate and agreed first.
- Do not claim application counts, competition, employer intent, time-to-fill or hiring probability.
- Do not present automated or model-generated judgments, passing tests or plausible-looking results as validated recommendation quality. The ranker comparison has not been run.
- Nothing is sent to an outside AI service unless the person whose text it is agrees first. Jacob's resume has not been sent anywhere.
- Keep source data, profiles with personal information, credentials (`.env`) and generated artifacts out of Git. Share data only privately; the NDA forbids public posting.
- Count listings separately from unique employers; duplicates, reposts and the companies-file join remain incompletely measured.

## Evidence Index

- [APP.md](APP.md) (what's left to do), README step 12 (how to run), and [docs/ml-inventory.md](docs/ml-inventory.md) (every ML piece, its validation, what not to claim)
- [Hiring signals report](docs/hiring-signals.md) and [what the first/last-seen dates mean](docs/signal-semantics.md)
- [Corpus report](data/matching/ai_ml_corpus_v1/corpus_report.json) and [classifier results](data/matching/role_family_classifier_20261009T205400Z/metrics.json)
- [Journal and experiment chronology](docs/archive/journal.md) and [data artifacts](docs/artifact-index.md)
- [Data feasibility summary](docs/discovery-summary.md) and [data preparation](docs/data-preparation.md)
- [Matching implementation notes](docs/job-matching-engine.md)
- [Grounded-explanation contract](docs/grounded-match-explanations.md) and [API contracts](docs/api-contracts.md)
- [Backend API](docs/backend-api.md) and [frontend/backend map](docs/archive/frontend-backend-map.md) (the map is partly stale)
