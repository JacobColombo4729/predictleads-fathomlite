# Machine learning in the app: what it is, why, and how well it is validated

A plain inventory for defending the project. For each piece: what it does, the principle behind it, why we chose it, what evidence we have, and what to say if challenged. "Validated" means measured against something outside the model itself.

## At a glance

| Piece | Where it shows up | Kind | Trained by us? | Validated? |
|---|---|---|---|---|
| 1. Job ranker (TF-IDF + BGE, rank fusion) | Discover | Retrieval: sparse + dense embeddings | No (pretrained BGE) | **Role level, ML_Fit's version: yes, on real career moves** (RRF 70.0% top 3 vs. 63.2% baseline); the app's own listing ranker: scoring planned |
| 2. Relevance feedback (👍/👎) | Discover | Rocchio query adjustment | No | No |
| 3. Compass analysis | Compass | Embeddings + clustering + similarity matching + MMR | No | No (thresholds hand-tuned on a few examples) |
| 4. Magic fill (resume to profile) | Profile | GenAI (LLM) with grounding checks | No (hosted model) | Checked by hand on two real resumes |
| 5. Hiring signals | Signals, job-card badge | Anomaly detection + entity resolution, backtested | Models tried, not shipped | **Yes, temporal backtest** |
| 6. Role-family classifier | Not in the app | Keyword (TF-IDF) + logistic regression on job titles; BGE title embeddings tried | Yes | Yes, against provider labels (about 72% agreement; not against fit) |
| 7. Match explanations | Off by default | GenAI with grounding | No | Not run in the demo |
| 8. Ask the market | Signals tab | GenAI turns a question into a validated search; the answer is plain data analysis | No (hosted model) | Checked by tests and live on public questions; the engine's numbers are counts and medians of real listings |
| 9. Company research and voice | Signals tab | GenAI with web search (cited sentences only); speech-to-text | No (hosted model) | Tested on one company and one synthetic voice; not measured at scale |

## 1. Job ranker

- **What:** for a profile, score every eligible listing and show the best. Two retrievers run side by side: TF-IDF cosine (keyword match, so exact terms like "Salesforce" or "PyTorch" count) and BGE cosine (`bge-base-en-v1.5`, a pretrained text-embedding model, 768 dimensions, so "customer-facing demos" can match "sales engineering"). Their rankings are combined with **reciprocal-rank fusion** (depth 50, k=60, equal weight).
- **Principle:** hybrid retrieval. Lexical and semantic retrievers fail in different ways, and fusing ranks (not scores) needs no score calibration and no tuning. Reciprocal-rank fusion is a standard, parameter-light method from the information-retrieval literature (Cormack et al., 2009).
- **Why not train a model:** there are no listing-level match labels, and hiring outcomes are not in the data. Pretrained embeddings plus a transparent fusion rule are the honest option for a cold start. Real career moves do give role-level outcomes: ML_Fit's trained layer learns from them and they are the evaluation (below).
- **Evidence, against real outcomes with no labelling** (`analysis/STANDARDS.md`, Validation design): for 2,813 real career changers in the public résumé data we know the role each actually moved into. ML_Fit scores each method on whether that role is in its top 3 from the person's background alone, against the same most-common-role baseline, on held-out people with 95% intervals. Test: TF-IDF + BGE fused (RRF) 70.0% at top 3, against the baseline's 63.2%; ML_Fit's trained layer 81.7% [78.5%, 84.7%]. That is ML_Fit's role-level version of these retrievers; **the app's listing ranker itself is scored the same way next** (each test person's top listings turned into a role ranking; `analysis/ANALYSIS.md`, combined-model plan step 7). The blinded labelling review is no longer the plan for evidence; the review page stays only as an optional usability check, and it refuses the team's own profiles.
- **If challenged:** "On real career moves, fused TF-IDF + BGE finds the role people actually moved into more often than the most-common-role baseline (70.0% vs. 63.2% at top 3, held-out people), and the trained layer on top of it does better still (81.7%). The app's own ranker is being scored on the same answer key." Scores are retrieval orders, never match percentages or hiring probabilities.

## 2. Relevance feedback

- **What:** 👍 pulls the query toward the liked job (weight 1.0), 👎 pushes away (weight 0.5), applied to both the keyword and the embedding query. Rated jobs are hidden.
- **Principle:** Rocchio relevance feedback (1971), a classic and well-understood method.
- **Honest limits:** it adapts one session; nothing is retrained and ratings die with the session. The weights are conventional defaults, not tuned. Users only rate what the ranker already showed them (selection bias).

## 3. Compass

- **What:** from 3 to 20 saved jobs it finds themes, how the profile lines up, and other roles worth a look.
- **Methods:** BGE embeddings of the jobs; agglomerative clustering for themes (at most 3); requirement sentences clustered (cosine distance 0.22, at least 2 jobs); each requirement matched to the closest verbatim line of the person's profile by cosine (coverage from 0.50 to 0.80 mapped to 0 to 100); suggestions chosen by **maximal marginal relevance** (relevance minus redundancy, Carbonell and Goldstein, 1998).
- **Safeguards:** the evidence is always the person's own words, shown verbatim. Years, degree and certification requirements are never auto-marked "supported". Duplicate listings are collapsed.
- **Honest limits:** the thresholds were set by looking at a few real examples. They are heuristics, not validated. No outside AI call is made; everything runs locally.

## 4. Magic fill (resume to profile)

- **What:** a hosted LLM (`gpt-4.1-mini`, temperature 0) reads resume text and returns structured profile entries.
- **Safeguard:** every extracted item must carry a quote that actually appears in the resume; items whose quote cannot be found are rejected. It is opt-in and the resume is not stored.
- **Honest limits:** checked by hand on two real resumes, not measured at scale. Cost was capped.

## 5. Hiring signals (the one with a real backtest)

- **What:** per company, new openings in the last 30 days against the company's own prior 90-day pace, giving burst / sustained / cooling / quiet.
- **Methods:** z-score anomaly detection with a count-based variance; entity resolution (repost merging by company, title and description); rule-based flags for bulk-import days and crawler dropouts; no look-ahead at any cutoff.
- **Evaluation:** temporal holdout (train on earlier cutoffs, test on 2026-08-31, 23,961 companies). We also trained gradient-boosting models on the momentum features. **They did not beat "how many openings last month" for predicting volume**, so we did not ship them and we do not claim a forecast.
- **What it can claim:** bursts are a real, timing-sensitive signal. Flagged companies stay above their own baseline next month 59% of the time vs 16% for unflagged ones, but keep only about 64% of their volume (sustained hirers keep 99%).
- **Honest limits:** four months of data, one test month, dates are crawler observation times (not posting dates). See [hiring-signals.md](hiring-signals.md) and [signal-semantics.md](signal-semantics.md).

## 6. Role-family classifier (supporting, not shipped in the app)

- **What:** logistic regression predicting 17 classes (16 role families plus Other) from the job title (one variant also reads the first 2,500 characters of the description), trained on the provider's own O*NET occupation labels (no human labels). Company-held-out 80/20 splits, repeated over 3 random splits so a result comes with a spread.
- **Result (2026-10-13, 244,527 labelled listings, mean over 3 splits):** always guessing the biggest class 30.2% accuracy; TF-IDF on the title **72.1%** (macro-F1 0.64); BGE embedding of the title 70.5% (0.62); TF-IDF on title + description 68.9% (0.57). Spread between splits is about 0.5 points. Strongest families: Technical Sales / Solutions Engineer 86%, Data Scientist 80%. Weakest: QA / Test 27%, Systems Analyst 39%, Network / Systems Admin 43%, Web Developer 48%.
- **Correction (2026-10-13):** an earlier version reported 94.7% accuracy (macro-F1 0.94) for BGE + logistic regression against 69.4% for TF-IDF. That comparison was not valid. The saved BGE job text includes "Occupation: <O*NET occupation name>", and the labels come from that same name, so the model could read the answer. A control that deliberately adds the occupation name to the keyword model scores 96.4%; with it removed, the BGE model scored about 62% in a same-split check on 6,000 jobs. Model inputs now never include the occupation, and `tests/test_role_family_classifier.py` keeps it that way. **Do not quote 94.7%.**
- **What it suggests:** the provider's labels follow the job title closely. The title alone scores as well as or better than title + description, and embeddings did not beat keywords at the same data size. (A reading, not a proof: the provider's labelling method is not documented.)
- **Honest limits:** it measures agreement with the provider's labels, not whether a job fits a person. Some fits stopped at their iteration limit, so the numbers are approximate. The earlier resume-profile prediction step was removed: it applied a job-title model to resumes, was never validated, and described past career, not the target. The Signals layer uses rule-based role families, not this classifier.

## 7. Match explanations (off by default)

A grounded GenAI explanation of why a job fits, restricted to facts in the profile and listing. Built and tested offline, not part of the default demo.

## 8-9. Ask the market, company research and voice

- **What:** a question box over our hiring data ([ask-the-market.md](ask-the-market.md)). A hosted model only translates the question into a search; our code validates it, runs it on the listing table and shows how it was read. An optional step researches one company with web search, and a Talk button turns speech into text.
- **Why it is defensible:** the AI never writes the numbers. Places, technologies and companies must literally appear in the question or they are removed (and the screen says so). Forecasts and advice are refused by a rule, not by the model. A company synopsis keeps only sentences that carry a web citation, and shows its sources.
- **Honest limits:** the model can misread an unusual question (the read-back chips let a person see and fix that); web research can be wrong or out of date and says so; counts come from one crawler snapshot with crawler dates.

## What we should not claim

- That any ranker is more accurate than another (not measured).
- That scores are probabilities or match percentages.
- That bursts forecast future volume (the backtest says they taper).
- That listing dates are posting or closing dates.
- That the Compass thresholds are validated.

## Likely questions

- **"Why not fine-tune a model?"** There are no listing-level labels. Real career moves (2,813 people) give role-level outcomes: enough to train ML_Fit's small trained layer and to evaluate, but small for fine-tuning BGE (an optional later step in the combined-model plan, kept only if it beats the trained layer on the same test people).
- **"How do you know it works?"** Signals: temporal backtest with baselines. Roles: real career moves, held-out people, against a baseline (ML_Fit: 81.7% vs. 63.2% at top 3). Listings within roles: ML_Rank's backtest on a later month (precision@10 0.500 vs. 0.031 for newest first). The app's own ranker: scored on the same career moves next.
- **"Is this just ChatGPT?"** No. Ranking, feedback, Compass and Signals use no LLM. The LLM appears only in resume reading (opt-in, quote-checked) and the optional explanations.
