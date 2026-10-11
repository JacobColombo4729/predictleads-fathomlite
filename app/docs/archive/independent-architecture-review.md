# Independent Architecture Review

**Date:** 2026-10-09 · **Reviewer role:** fresh ML/backend reviewer · **Mode:** read-only; nothing in the project was changed except this file.

**How to read this:** **[Fact]** means I saw it in a saved file or in code. **[Hypothesis]** is my inference and is untested. **[Model]** means the claim comes from a model-generated judgment (the GPT-4.1 / GPT-4.1-mini graders), not from a person or the source data.

**What I inspected:** the six named documents; `backend/api.py`; `job_matching_engine.py`; `grounded_match_explanations.py`; the three *development* profiles in `code/weighted_rrf_profiles.json`; the weighted-RRF, hybrid and v3.1 qualification reports; the frozen grader prompt; and the dev top-five rankings and grade-ceiling diagnosis.

**Held-out boundary:** I did not open the six unopened held-out profiles, `heldout_profiles.json`, `heldout_evaluation_pending.jsonl`, or any held-out results. Two caveats, for transparency:
- `profile_split_manifest.json` lists their IDs (names only), and I read that.
- My first profile-reading script loaded `weighted_rrf_profiles.json` into memory. It crashed before printing anything from the held-out key.

I used no credentials, ran no provider call, embedding, ranking, test or experiment, and did not touch the ledger, branches or commits.

---

## 1. What is implemented, evaluated, connected, and missing

| Area | Implemented | Evaluated | Connected to the product | Missing |
|---|---|---|---|---|
| Data prep | Prepared-record schema; 30,000-job random sample (24,514 with usable descriptions) from a 13.4M-listing audit | Audit completed to end of file [Fact]; extraction spot-check of 10 descriptions (97% category agreement, but 18 possibly missed lines) [Model] | Backend loads the sample | Duplicate/repost handling; companies-file join; any check that the sample contains enough *relevant* jobs per niche transition |
| Matching | Keyword, TF-IDF (word-weighted text similarity), MiniLM and BGE embeddings (models that turn text into numeric "meaning" vectors), RRF (reciprocal-rank fusion: merges two ranked lists by position), cross-encoder (a model that reads profile and job together to rescore) | Only by GPT-graded synthetic profiles, 2–5 usable profiles per comparison [Model] | **TF-IDF only.** The backend has no BGE or RRF code (`grep` finds none) | Serving path for the method the team calls its reference (equal-weight TF-IDF+BGE RRF); a BGE job-vector cache the API can load |
| Explanations / skills gap | `grounded_match_explanations.py` with evidence-ID validation | 12 offline tests with a mocked provider [Fact]. **Never run live** | API endpoint exists but live generation is disabled by default | Any live quality check; any UI display |
| Backend | FastAPI demo: sessions, profiles, search, TF-IDF recommendations, assessment wrapper, saved jobs, SQLite | Its own tests (mock or tiny corpora) | **Not connected to the frontend** [Fact: no `fetch` or `/api/v1` anywhere in `frontend/src`] | Frontend client, real states, deployment |
| Frontend | Four mock screens with fictional fixtures | n/a | Fixtures and `localStorage` only | Everything in the real journey |
| Extra GenAI feature | Basic skills-gap projection inside the assessment | None | No | See explanations |

**Documentation discrepancies:**
1. `roadmap.md` (Oct 8) says "There is no backend web application". `docs/backend-api.md` and `docs/api-contracts.md` say one exists, and `backend/api.py` is 1,110 lines. The roadmap is stale on this point.
2. The roadmap names equal-weight RRF the working reference, but the only ranker a user could reach is TF-IDF.
3. The backend builds its ranking profile from title, organization, description, skills and desired roles, and sets `preferences={}`. The experiments used richer fictional profiles (work history, accomplishments, a transition goal, explicit absences and unknowns). **What was evaluated is not what would be served.**
4. In `development_rankings.csv`, rows labelled `score_type = tfidf_cosine` carry scores around 0.02–0.03. That is RRF scale, so the label looks wrong. A downstream consumer could misread these scores. [Hypothesis: a labelling bug; I did not trace the writer.]
5. The "Current state" paragraph at the top of the journal is dated Oct 7, and its "embeddings are provisional" framing is superseded by later entries.

## 2. Supported conclusions and open questions

**Supported [Fact]:**
- The pipeline runs end to end offline on the saved sample. Evidence-ID checks, budget ledgers and preserved raw outputs are good provenance practice.
- Truncation was real. MiniLM's 256-token limit cut 205 of 220 saved documents, and BGE still omitted text for 16,240 of 24,514 jobs. Input preparation is a legitimate concern.
- Citation-ID validation proves a quote exists in the source. It does **not** prove the model's interpretation is right.
- The qualification-protocol run shows this concretely. Six of 12 pairs failed technically (five hit the output-token cap). Five of the six that validated carry semantic concerns, such as turning a drinking-water-license absence into a wastewater-license contradiction, or turning "unknown" into a "gap". Only one pair is clean, so the 0/1 grade agreement means nothing.
- No ranking has human validation. No experiment supports a "winner".

**Plausible but not established:**
- RRF beats TF-IDF. In the hybrid run, P@5 was 0.36 vs 0.20 on five profiles, but that is about one extra good job per profile, graded by a model. The weighted-RRF dev run did not select a better weighting, so I read it as "no information", not as "equal-weight is best".
- "Strong fit" never occurs (no grade 3 in 45 dev top-five placements). I see three candidate causes and the data cannot separate them:
  1. The rubric. Grade 3 requires "no known material mismatch", which a career changer by definition rarely meets.
  2. The sample. A 0.22% random draw may hold few ideal jobs for niche transitions such as water operations.
  3. The ranker.
- **Concrete evidence the grader is unreliable on the very thing the project cares about:**
  - In the saved hotel-profile judgment, `material_gaps` lists "No forklift certification…". The profile lists forklift as *unknown*, and the frozen prompt itself says a missing qualification is unknown, not unmet.
  - The grade-ceiling diagnosis also finds "Collections Associate" (accounts-receivable collections) at rank 5 for a museum-collections profile. That is a plain title collision that survived into the top five.

## 3. Fit with the career-transition goal

- **Data preparation: reasonable.** Raw and derived fields are kept side by side, unknown is not treated as open, and the historical caveat is honoured. The weak spot is coverage. A uniform random 30k sample will thin out exactly the rare roles a career changer might discover. I would check that before drawing conclusions about the ranker. [Hypothesis]
- **Candidate profiles: suitable for smoke testing, weak as evidence.**
  - Strengths: they are fictional, privacy-safe, and encode explicit absences and unknowns, which is the right structure for gap testing.
  - Weaknesses: they are tidy, authored by the same AI-assisted process, and each names `desired_roles` titles close to the target jobs. TF-IDF triple-weights those titles, so the "unfamiliar title" challenge is partly pre-answered.
  - Real users write messy resumes with no target title.
  - Each comparison reaches only 2–5 usable profiles, which is too few to separate methods.
- **Ranking:**
  - The sequence of ideas (truncation, then section-aware text, then hybrid) is sensible.
  - Two decisions I'd contest:
    - Long CPU-encoding cycles (about 2 hours per full corpus encode) were spent on variants whose effect could not be detected at this sample size.
    - The cross-encoder is out-of-domain; the team already benched it.
- **Evaluation: not suitable as built.**
  - The rubric measures "how qualified is this person", but the goal is "is this a realistic job worth exploring". A good transfer pick with two learnable gaps is graded 2, which is acceptable but not distinguishable from a mediocre pick.
  - The grader is a model that conflates unknown with gap, and that behaviour persisted across three prompt revisions and two models.

## 4. Too much effort on evaluator machinery? Yes.

Roughly the last third of the journal and about eight scripts (`run_weighted_rrf_grader_audit`, `..._qualification_protocol`, `..._model_comparison`, `analyze_*`, `evaluate_*`) went into making a model grader trustworthy, and it still isn't. About $1.57 of usage ($2.92 reserved of the $3 cap) is spent. Almost no budget remains, and a person has not yet judged a single pair.

**Simplest defensible approach:**
1. Two team members label the top five from each method for 8–10 profiles on a three-level scale:
   - **Worth exploring** (credible duty transfer; gaps are learnable or unknown)
   - **Maybe**
   - **Not realistic** (wrong duties, title collision, or a hard requirement the profile clearly fails)
2. Add one line of reason per label, with **Unknown** allowed.
3. Double-label 20% to report simple agreement.
4. Report per-profile counts ("3 of 5"), win/tie/loss between methods, and a short list of failures. Do not report averages as if they were significant.
5. Use model grades only as a triage hint, never as truth.

This is an hour or two per person and costs $0.

## 5. Retain, simplify, defer, change

- **Retain:** provenance habits (frozen plans, preserved raw outputs); the unknown/gap distinction; the "historical, not verified open" labelling; the ranker interface in the backend; the evidence-catalog idea in explanations.
- **Simplify:**
  - Freeze the evaluator work. Don't build a smaller response contract (the current handoff direction). Two models and four protocol versions point to the task, not the schema, as the problem.
  - Keep the Python-side eligibility recalculation idea (derive status deterministically, with the model only proposing excerpts), but only if explanations need it.
  - Collapse the experiment runners into one reusable "rank, pool, label" workflow.
- **Defer:** cross-encoder; further weighting experiments; the six held-out profiles (keep them sealed until a human protocol exists); the Skill Opportunity Optimizer, Career Adjacency, timing models and market momentum.
- **Change:**
  1. Add the equal-weight RRF ranker to the backend. This needs the BGE vector cache, a job-ID mapping, and a profile builder that sends the same fields the experiments used. Fix the `score_type` labelling.
  2. Connect the frontend to the existing profile → recommendations → job detail path.
  3. Make explanations safe by design: show job excerpt next to profile excerpt, state "not evidenced in your profile" rather than "gap", and restrict model output to suggested duty pairings. The evaluator history shows the same model family will invent contradictions even with valid citation IDs. Run the explanation once live on about 10 pairs and read every output before showing it in a demo.
  4. Revise the rubric for human use, so "realistic stretch with learnable gaps" can be the top level.

## 6. Single recommended next step

**Do a $0 human labelling of existing saved pools, using the three-level scale above.** Start with the hybrid pool (120 pairs, TF-IDF vs RRF vs cross-encoder, five complete profiles) and the 39 weighted-RRF dev pairs. Two people, about two hours each, with shuffled and blinded method labels.

**How to test whether it helped (set the rules before labelling):**
1. **Ranker choice.** If humans find RRF at least as good as TF-IDF on at least four of the five hybrid profiles (counting "Worth exploring" in the top five), wire RRF into the backend. If not, ship TF-IDF and stop investing in embeddings for the demo.
2. **Grader value.** Compare human labels to the GPT grades. If agreement on "not realistic" vs the rest is below about 70%, retire model grades as a metric. If it is higher, keep them as a cheap triage signal only.
3. **Rubric and sample ceiling.** If humans do rate top-five jobs "Worth exploring" where the model never gave a 3, the ceiling is the rubric, not the ranker. If humans also find few good options, suspect the sample, and check coverage with a keyword search on the full file for one niche transition.

The result is an honest, small, human-anchored answer in an afternoon. It also tells us which ranker to wire up, which is the biggest remaining product gap.
