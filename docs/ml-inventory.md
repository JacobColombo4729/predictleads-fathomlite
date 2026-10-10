# Machine learning in the app: what it is, why, and how well it is validated

A plain inventory for defending the project. For each piece: what it does, the principle behind it, why we chose it, what evidence we have, and what to say if challenged. "Validated" means measured against something outside the model itself.

## At a glance

| Piece | Where it shows up | Kind | Trained by us? | Validated? |
|---|---|---|---|---|
| 1. Job ranker (TF-IDF + BGE, rank fusion) | Discover | Retrieval: sparse + dense embeddings | No (pretrained BGE) | **No relevance result yet** |
| 2. Relevance feedback (👍/👎) | Discover | Rocchio query adjustment | No | No |
| 3. Compass analysis | Compass | Embeddings + clustering + similarity matching + MMR | No | No (thresholds hand-tuned on a few examples) |
| 4. Magic fill (resume to profile) | Profile | GenAI (LLM) with grounding checks | No (hosted model) | Checked by hand on two real resumes |
| 5. Hiring signals | Signals, job-card badge | Anomaly detection + entity resolution, backtested | Models tried, not shipped | **Yes, temporal backtest** |
| 6. Role-family classifier | Not in the app | Logistic regression on BGE | Yes | Yes, against provider labels (not against fit) |
| 7. Match explanations | Off by default | GenAI with grounding | No | Not run in the demo |

## 1. Job ranker

- **What:** for a profile, score every eligible listing and show the best. Two retrievers run side by side: TF-IDF cosine (keyword match, so exact terms like "Salesforce" or "PyTorch" count) and BGE cosine (`bge-base-en-v1.5`, a pretrained text-embedding model, 768 dimensions, so "customer-facing demos" can match "sales engineering"). Their rankings are combined with **reciprocal-rank fusion** (depth 50, k=60, equal weight).
- **Principle:** hybrid retrieval. Lexical and semantic retrievers fail in different ways, and fusing ranks (not scores) needs no score calibration and no tuning. Reciprocal-rank fusion is a standard, parameter-light method from the information-retrieval literature (Cormack et al., 2009).
- **Why not train a model:** there are no match labels. Hiring outcomes are not in the data. Pretrained embeddings plus a transparent fusion rule are the honest option for a cold start.
- **Evidence:** none for relevance. The blinded review (Jesse and Jacob labelling pairs) has not been done, so TF-IDF vs BGE vs RRF is **not compared on this audience**. A recall check against the target roles you wrote before looking at rankings exists but is deliberately unread until labelling is finished.
- **If challenged:** "This is a defensible default, not a proven winner. Here is the plan to measure it, and the labelling page is built." Scores are retrieval orders, never match percentages or hiring probabilities.

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

- **What:** BGE embeddings + logistic regression predicting 17 role families, trained on the provider's own O*NET occupation labels (no human labels), with a company-held-out 80/20 split so one company's wording cannot leak.
- **Result:** test accuracy 94.7% (macro-F1 0.94) vs 69.4% for TF-IDF + logistic regression and 29.8% for always guessing the biggest class.
- **Honest limits:** it measures agreement with the provider's labels, not whether a job fits a person. On resumes it describes past career, not the target. The Signals layer uses rule-based role families, not this classifier.

## 7. Match explanations (off by default)

A grounded GenAI explanation of why a job fits, restricted to facts in the profile and listing. Built and tested offline, not part of the default demo.

## What we should not claim

- That any ranker is more accurate than another (not measured).
- That scores are probabilities or match percentages.
- That bursts forecast future volume (the backtest says they taper).
- That listing dates are posting or closing dates.
- That the Compass thresholds are validated.

## Likely questions

- **"Why not fine-tune a model?"** No match labels exist. The blinded review would produce a few hundred; that is enough to evaluate, not to train.
- **"How do you know it works?"** Signals: temporal backtest with baselines. Ranker: not yet; the labelling step is the planned answer.
- **"Is this just ChatGPT?"** No. Ranking, feedback, Compass and Signals use no LLM. The LLM appears only in resume reading (opt-in, quote-checked) and the optional explanations.
