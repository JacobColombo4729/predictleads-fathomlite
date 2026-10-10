# PredictLeads Career Intelligence Roadmap

Updated: October 9, 2026  
Status: Working roadmap, rewritten to match the connected demo. For the live state of each piece, [STATUS.md](STATUS.md) is the source of truth and wins over this file if they differ.

**Audience:** career changers, starting with ourselves (TripleTen AI/ML graduates), looking for realistic roles in AI/ML, data, and technical sales / solutions engineering.  
**Target date:** about 2026-10-23. Deployment target, owners and demo format are not yet decided.

## Product Goal and User Journey

Help a career changer find relevant listings, understand the evidence, and see what to do next.

**Profile → Discover (ranked jobs) → job details → gap analysis → saved jobs**

1. **Profile** (built): blank start; fill the form or use **Magic fill** on a PDF/text resume (GenAI feature 1); entries are typed Job / Project / Education; add desired roles; Save.
2. **Discover** (built): the hybrid ranker orders about 81,000 historical listings against the saved profile; filters; 👍/👎 relevance feedback adjusts the list live.
3. **Job details** (built, basic): description, source link, historical-snapshot labelling, and a keyword-based skills check.
4. **Gap analysis for a selected job** (GenAI feature 2, coded but not connected; see section 5).
5. **Saved jobs** (built): stages and notes.

Status terms: **Built** means working in the running app and verified; **Implemented** means code exists but is not integrated or evaluated; **Experimental** means bounded evidence, not validated quality; **Planned** means no implementation yet.

## 1. Data and Candidate Profiles

**Built**
- Full snapshot scan: 13,415,388 job listings read to end of file with no errors (snapshot 2026-09-30). A targeted pool for AI/ML, data and technical-sales roles has 319,970 matches (`code/extract_targeted_pool.py`).
- Ranking corpus: 244,558 unique listings (targeted pool plus a random 30,000 background sample), usable descriptions only, exact duplicates collapsed (17,991). 163,274 listings are explicitly `closed` and excluded; the **81,284 listings with unknown status** form the serving set. See [corpus report](data/matching/ai_ml_corpus_v1/corpus_report.json).
- Candidates: 10 distinct candidates = 12 profiles (2 real and consenting, Jesse and Jacob, each with and without target titles; 8 fictional). Real resumes and profile text stay in the Git-ignored `data/` folder.
- Time fields: `first_seen_at` / `last_seen_at` are kept as raw and parsed values and described as an **observation window** (median 24 days across the corpus). They are not posting or closing dates; openness is never inferred from dates or null status. No time-based feature is built.

**Open**
- Companies are only the same-wrapper company reference on each listing; the separate companies-file join is not validated.
- Duplicates are collapsed only when company, title and description are identical; reposts with edited text remain.

## 2. ML Matching

**Built: the demo ranker**
- Equal-weight reciprocal-rank fusion (depth 50, k=60) of TF-IDF cosine and BGE (`bge-base-en-v1.5`, 768 dimensions) cosine. Listing vectors were computed on GPU; profile queries are encoded on CPU inside the backend. Scores are retrieval signals, not match percentages or hiring probabilities.
- Relevance feedback: each 👍/👎 moves both the keyword and embedding query (Rocchio-style) toward liked and away from rejected jobs, and hides rated jobs. It adapts one demo session; the model is not retrained and ratings are deleted with the session (24-hour TTL or Start over).
- Baselines: keyword overlap and TF-IDF cosine live in `code/job_matching_engine.py` (command line). The backend can also serve TF-IDF alone with `.\run_demo_backend.ps1 -Ranker tfidf`.

**Not validated.** The planned blinded human review (139 pairs, page built) was dropped for time, so **no relevance result exists for any ranker** and TF-IDF vs BGE vs RRF has not been compared on this audience. RRF is the working reference because it is the stronger-looking option and its top results looked sensible. The write-up and demo must say this plainly.

**Built: role-family classifier (supporting model, not in the app)**
- BGE embeddings + logistic regression predicts 17 role families from the provider's O*NET occupation label (no human labels), with a company-held-out 80/20 split. Test accuracy 94.7% (macro-F1 0.94) against 69.4% for TF-IDF + logistic regression and 29.8% for the majority class. This measures agreement with provider labels, not fit. On resumes it describes past career rather than target roles, so it is not used to rank or filter. See `code/role_family_classifier.py` and `data/matching/role_family_classifier_20261009T205400Z/`.

**History, not evidence.** Earlier experiments (MiniLM, section-aware text, cross-encoder, weighted RRF, faceted text) used synthetic profiles in other job areas and model-generated grades, and were small and inconclusive. They are kept as history in [docs/journal.md](docs/journal.md); do not quote them as evidence for the final product.

## 3. GenAI Features

**Feature 1: resume "Magic fill" — Built and live-tested**
- Upload a text-based PDF, `.txt` or `.md` resume (PDF read locally with `pypdf`; scanned PDFs are rejected). Emails, phone numbers and links are removed, then the text is read by OpenAI (`gpt-4.1-mini`, temperature 0, strict JSON schema) after a per-use consent checkbox. It is off unless the backend starts with `.\run_demo_backend.ps1 -ResumeFill`.
- Grounding: every experience, project, education and skill item must carry a quote found verbatim in the resume or it is dropped and counted. Desired roles are never extracted. A fill replaces name, headline, location, experience and skills. Skill "evidence" lines are the model's attribution and are not covered by the quote check, so users should review them.
- Tested on Jesse's resume (9 entries, about 18-24 skills, nothing dropped, wording varies slightly run to run); 26 offline tests. Budget: the original $3 cap is used up; an extra $2 was approved on 2026-10-09 and only a few cents have been spent.

**Feature 2: grounded gap analysis for a selected job — Implemented, not connected**
- `code/grounded_match_explanations.py` and `POST /api/v1/profiles/{id}/jobs/{job_id}/assessment` exist and are tested offline. The assessment lists the job's requirements as required/preferred/unclear, marks each supported / explicit gap / unknown with cited evidence, lists transferable experience, and suggests next steps. The provider is disabled by default, and the UI does not call the endpoint; job details show only the keyword skills check.
- Planned design: **retrieve** the most relevant profile evidence for each job requirement with the BGE embeddings; **judge** with the LLM only on that evidence, with the existing citation validation; **plan** next steps for unknown and gap items. Run on click only, then cache.
- Rules: missing profile information is "not mentioned in your profile", never a gap. An explicit gap needs directly evidenced contradiction. No claims about hiring, eligibility or becoming qualified; do not invent courses or certificates. Needs your explicit consent per use and a budget decision before live calls.

## 4. Backend and Frontend

**Built**
- FastAPI backend (`backend/api.py`): anonymous cookie sessions in SQLite, profile create/read/update with version checks, search, job and company details, recommendations (RRF or TF-IDF), feedback, saved jobs, assessment endpoint (provider off by default), resume extraction, and session reset. 46 backend tests pass.
- React + Vite frontend (`frontend/src`) wired to the API: Discover, Saved, Profile (Magic fill, entry types, blank start, **Start over** that deletes the server session). It starts from a blank profile, Discover says "Create your profile to see matches" until one is saved, and the right-hand panel shows a placeholder until then.
- Run locally: `.\run_demo_backend.ps1 [-ResumeFill]` (about 3.5 minutes to load, about 2.5 GB memory) and `cd frontend; npm run dev`, then open http://127.0.0.1:5173.

**Known gaps**
- The frontend README and [docs/backend-api.md](docs/backend-api.md) still describe the earlier mock and TF-IDF-only state; refresh them.
- "Skills & Paths" (milestone 5 of the UI roadmap) is not built.
- Sessions are anonymous and temporary: no accounts, no persistence across visits.
- The frontend has no automated tests; only type-checking and manual use.

## 5. Remaining Work, in Order

1. **Checkpoint in git.** Nothing from the connected demo is committed; the branch is `feature/job-matching-engine-baseline`.
2. **Connect gap analysis** (feature 2): job-details button, evidence and unknown states, failure states, then a bounded live test on Jesse's resume and a few jobs he knows well.
3. **End-to-end rehearsal** on a real profile with desired roles filled in; fix what breaks.
4. **Clean notebook** for the final path (data, ranker, classifier) and the **write-up**, with measured versus unvalidated claims stated plainly.
5. **Demo plan**: live or recorded, plus a recorded fallback; decide whether to deploy or demo locally.
6. **Refresh stale docs** (frontend README, backend-api, artifact index) and update this file as items land.

## 6. Week 1 Deliverable Map (course roadmap)

| Requirement | Status |
|---|---|
| Explore and clean the job dataset | Done |
| Represent `first_seen_at` / `last_seen_at` over time | Decided: observation window; no feature built |
| Define the candidate profile and matching problem | Done |
| Decide how profiles are sourced | Done (2 real, 8 fictional) |
| Define evaluation metrics | Defined (precision@k, nDCG, blinded review); ranker evaluation not run |
| Baseline ML matching/ranking model | Done (TF-IDF baseline; BGE and RRF built on it) |
| Initial evaluation results | Classifier measured; ranker comparison not run |
| Define GenAI use cases | Done (Magic fill built; gap analysis coded) |

## 7. Nice to Have and Optional Research

**Per-user preference model ("Refine my matches") — nice to have; DECISION PENDING (added 2026-10-09)**
- Idea: after about 20-30 👍/👎 ratings, a **Refine my matches** button trains a small model on that user's ratings and re-ranks their results. Today's feedback is only a live Rocchio-style nudge inside one session; the model itself is not retrained.
- Planned design: unlock at about 10 ratings with at least 5 of each kind (labelled "low confidence", "reliable" from about 30); fit a heavily regularized logistic regression or ridge on the BGE vectors of the rated jobs, on demand; cross-validate on the user's own ratings and refuse to apply the model if it does not beat chance; score all eligible listings and fuse that rank with the RRF; show "refined with N 👍 and M 👎, predicted your ratings with X% accuracy, K of your top 10 changed"; keep some variety to avoid a filter bubble. Optional: reason chips on 👎, an editable "you seem to prefer / avoid" list.
- Cost: all local; about half a day. How many ratings are enough is unknown until measured: a learning curve (a user rates about 40 jobs; train on the first 5, 10, 15 and so on; test on the rest) answers it.
- Prerequisite: an **opt-in rating export**, which also supplies the TF-IDF vs BGE vs RRF comparison that was dropped.
- Risks: small samples and noisy accuracy; selection bias (users only rate what the ranker already showed); ambiguous 👎; no accounts, so no persistence; storing a taste model next to resume content needs consent. Cross-user learning needs far more users and is future work.
- Revisit and decide go/no-go after the demo, gap analysis and write-up.

**Other optional work (as time permits)**
- **Skills across top jobs:** run the gap analysis over a user's top or saved jobs and show the most frequent missing skills (the unbuilt Skills & Paths idea). A fuller Skill Opportunity Optimizer needs skill normalization and a defensible opportunity population; report association, not a guarantee of qualification.
- **Career Adjacency:** a distinct, interpretable feature would need exposed transferable duties and validated O*NET assignments; matching transitions is not a graph.
- **Listing timing:** descriptive first-to-last-seen summaries with caveats are possible; survival or hiring-time modelling is unsupported by one snapshot.
- **Market momentum:** needs comparable snapshots; not possible from one.
- **PDF OCR, accounts and persistent profiles, deployment hardening:** not planned for the demo.

## Preserved Decisions and Limitations

- Keep missing, null, blank, invalid and unknown distinct. Do not infer that a listing is open from null status or from dates; every listing is a historical record from the 2026-09-30 snapshot.
- Do not claim application counts, competition, employer intent, time-to-fill or hiring probability.
- Do not present automated or model-generated judgments, passing tests or plausible-looking results as validated recommendation quality. The ranker comparison has not been run.
- Nothing is sent to an outside AI service unless the person whose text it is agrees first. Jacob's resume has not been sent anywhere.
- Keep source data, profiles with personal information, credentials (`.env`) and generated artifacts out of Git.
- Count listings separately from unique employers; duplicates, reposts and the companies-file join remain incompletely measured.

## Evidence Index

- [STATUS.md](STATUS.md) (current state, how to run, pending items)
- [Corpus report](data/matching/ai_ml_corpus_v1/corpus_report.json) and [classifier results](data/matching/role_family_classifier_20261009T205400Z/metrics.json)
- [Journal and experiment chronology](docs/journal.md) and [data artifacts](docs/artifact-index.md)
- [Data feasibility summary](docs/discovery-summary.md) and [data preparation](docs/data-preparation.md)
- [Matching implementation notes](docs/job-matching-engine.md)
- [Grounded-explanation contract](docs/grounded-match-explanations.md) and [API contracts](docs/api-contracts.md)
- [Backend API](docs/backend-api.md) and [frontend/backend map](docs/frontend-backend-map.md) (both partly stale; see section 4)
