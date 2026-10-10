# Project Status (start here)

**Direction (decided 2026-10-09):** help career changers, starting with ourselves (TripleTen AI/ML graduates), find realistic roles in AI/ML, data and technical sales/solutions engineering. Deadline: about 2026-10-23.

**Read this file first.** `docs/journal.md` and the older experiment folders are an archive of how we got here. They are accurate as history but are not the current plan. Earlier exploratory results (water operations, logistics, records control and so on) say nothing about AI/ML candidates and should not be quoted as evidence for the final product.

## Run the app (live, 2026-10-09)

1. **Backend** (about 3.5 minutes to load 81,284 listings, about 2.5 GB of memory): in PowerShell from the project folder, `.\run_demo_backend.ps1`. It serves `http://127.0.0.1:8000` with the TF-IDF + BGE (RRF) ranker. Use `-Ranker tfidf` for the lighter keyword-only ranker.
2. **Frontend**: `cd frontend; npm run dev`, then open `http://127.0.0.1:5173`.
3. **What you can do**: edit your profile (or import a profile JSON file, for example your own from `data/candidates/`), see ranked jobs, mark jobs "Good match" / "Not for me" and watch the list adapt, save jobs with stages and notes, and read a rule-based skills check per job ("not evidenced in your profile" means unknown, not missing).
4. The app now starts with a blank profile (no example person). Discover shows "Create your profile to see matches" until a profile is saved. "Start over (blank profile)" on the Profile page deletes the server session (profile, saved jobs, ratings) so the demo can be run again; a private browser window also gets a fresh session.

**Relevance feedback (built):** each rating moves the search toward liked jobs and away from rejected ones (Rocchio-style, applied to both the keyword and the embedding query), hides rated jobs, and shows "More like ..." notes. It adapts one demo session; it does **not** retrain the model, and it has **not** been shown to improve relevance. Recommendation calls take about 2.7 s.

**Ranker choice:** the demo uses equal-weight RRF because it is the working reference and its top results looked sensible. It is **not** validated: the planned human review was dropped for time, so no relevance result exists for any ranker. Say so plainly in the demo. The review page and recall check are still available if you later want evidence.

## Optional: human review page (not required)

The earlier blinded review page is still there if you want evidence later. Open `data/matching/ai_ml_rankings_20261009/review/review.html`. (Details below.)

## Where to open the review page

`data/matching/ai_ml_rankings_20261009/review/review.html` (double-click it, or open it in a browser from the file path; nothing is uploaded).

- Pick your name (Jesse or Jacob) in the top bar. It shows only your assigned pairs (82 each; 139 pairs in total, 25 labelled by both of you for an agreement check). Use the same way of opening the file every time: labels are saved in that browser, so press **Export my labels** when you stop and keep the file.
- Method, rank and score are hidden and the order is shuffled. The answer key is `.../review/answer_key.json`: do not open it until you have both finished.
- After you both export (`labels_jesse.json`, `labels_jacob.json`): `python code/analyze_review_labels.py --run-dir data/matching/ai_ml_rankings_20261009 --labels labels_jesse.json labels_jacob.json`.
- Before you look at any rankings, fill in `data/candidates/target_roles_TO_FILL_IN.md`; then `python code/recall_check.py --run-dir data/matching/ai_ml_rankings_20261009 --profile-id <id> --titles "<role>" ...`.

## What is current

| Piece | File | State (verified 2026-10-09) |
|---|---|---|
| Targeted job pool (AI/ML, data, technical sales) from the full snapshot | `code/extract_targeted_pool.py` | **Complete.** 13,415,388 listings scanned to end of file, 0 errors, 319,970 matches (about 36 min). Output: `data/matching/targeted_pool_20261009T164007Z/` |
| Ranking corpus: targeted pool + random 30k background, usable descriptions only, duplicates collapsed | `code/rank_candidates.py build-corpus` | **Complete.** 244,558 unique listings (220,779 from the targeted pool, 24,280 random background, 709 in both). 86,712 excluded for no usable description; 17,991 duplicate listings collapsed (see `duplicate_groups.jsonl`). Raw HTML field omitted from this file to save memory (originals remain in the pool/sample files). Output: `data/matching/ai_ml_corpus_v1/` |
| BGE embeddings | `code/rank_candidates.py encode` / `finalize-encode` | **Complete and verified.** 60 chunks, 244,558 x 768 float32, all finite, unit length. Merged into `bge_embeddings.npy`. Row order tied to the corpus by ordered job-ID hash; 60 randomly sampled vectors re-encoded and matched (min cosine 0.99999994). |
| Ranking: TF-IDF, BGE, equal-weight RRF per profile | `code/rank_candidates.py rank` | **Complete.** 12 profiles, top 10 per method, plus top 50 lists. Frozen settings: depth 50, RRF k=60, weights 1:1. Output: `data/matching/ai_ml_rankings_20261009/` |
| Blinded human review page | `code/build_review_page.py` | **Built, loaded in a browser, no console errors.** 139 pairs (top 5 of each method per profile, deduplicated; 41 pairs were returned by more than one method). |
| Review analysis | `code/analyze_review_labels.py` | Built; waiting for labels. |
| Recall check | `code/recall_check.py` | Built; needs your target-role lists (written before looking at rankings). |
| Candidates | `data/candidates/real_candidate_profiles.json` (private, ignored), `code/ai_ml_candidate_profiles.json` | **10 distinct candidates = 12 profiles**: 2 real (Jesse, Jacob) plus 8 fictional. Each real person has two versions (named target roles / no titles), which is why it is 12 profiles. Four profiles have no target titles (2 real variants, 2 fictional). Real profiles were hand-structured from the resumes; contact details left out; "desired roles" on the real profiles are drafts for you to edit. |
| API ranker (TF-IDF default, RRF opt-in) with relevance feedback | `backend/rrf_ranker.py`, `backend/feedback_ranking.py`, `backend/api.py` | **Running on the real serving corpus.** 20 offline tests pass; verified live over HTTP. |
| Frontend wired to the API | `frontend/src` (`lib/api.ts`, `state/useLive.ts`) | **Live.** Profile, recommendations, rating buttons, details, skills check, saved jobs; fixture data removed. Builds cleanly. |
| Serving corpus for the backend | `data/matching/ai_ml_serving_v1` (`rank_candidates.py export-serving`) | 81,284 listings with unknown source status plus their vectors. |
| Resume "Magic fill" (GenAI feature 2) | `code/resume_profile_extraction.py`, `POST /api/v1/profile-extraction`, Profile page block; start the backend with `.\run_demo_backend.ps1 -ResumeFill` | **Built and live-tested 2026-10-09** on Jesse's resume (9 experience entries, 23 skills, 0 dropped) and a fictional resume. Off by default; needs consent checkbox per use and OPENAI_API_KEY in `.env`. Accepts text-based PDFs (read locally with `pypdf`, installed in the ML `.venv`; 2 MB / 10 pages max; scanned image-only PDFs are rejected) and .txt/.md. Emails, phone numbers and links are removed before any text is sent to OpenAI. Each profile entry has a type (Job / Project / Education) chosen by a dropdown on the Profile page and pre-set by Magic fill; it only changes how entries are grouped on screen, matching still reads every entry the same way. A fill replaces name, headline, location, experience and skills outright. Skill evidence text is the model's attribution (for example "used in the HPE role") and is not covered by the verbatim-quote check, so review it. Every extracted item needs a verbatim quote found in the resume or it is dropped and counted; desired roles are never extracted. 24 offline tests pass; PDF path live-tested on Jesse's PDF (9 experience entries, 25 skills, 0 dropped). Model gpt-4.1-mini; the live tests were a handful of small calls (a few cents) against the extra $2 budget approved today. |
| Role-family classifier (supporting model, not in the app) | `code/role_family_classifier.py`; output `data/matching/role_family_classifier_20261009T205400Z/` | **Run 2026-10-09.** 17 classes from the source O*NET occupation (no human labels); company-held-out 80/20 split. Test accuracy: majority class 29.8%, TF-IDF + logistic regression 69.4%, BGE + logistic regression 94.7% (macro-F1 0.94). Measures agreement with the provider's labels, not fit. On resumes it describes past career, not the target (Jesse -> Sales Representative, Jacob -> Data Scientist, career changers -> Other), so it is not used as a fit predictor. |

Smoke-test outputs (tiny, in a temporary scratch folder, not under the project) were used only to test the code; they are not results.

## Eligibility in the rankings

All 12 profiles exclude listings whose source status is explicitly `closed` (163,274 of the 244,558 corpus listings). The remaining **81,284** have **unknown** status (null in the source). Unknown is not "open": every card in the review page says "status in snapshot: unknown (not verified as open)" and "historical listing from 2026-09-30". Location is not a hard filter.

## What is not decided

- Whether RRF really beats TF-IDF for this audience. The demo uses RRF by default; no relevance results exist for this direction.
- Job activity over time (decided 2026-10-09, no feature built): represented as an observation window (`first_seen_at` to `last_seen_at`, median 24 days across the corpus), kept as raw and parsed values. It is evidence of when a listing was observed, not the posting or closing date; openness is never inferred from dates or null status. A "recently observed" feature can be added later.
- No model-generated relevance grades are used. They were unreliable (they treated "not mentioned" as "gap").

## Known limitations

- Title/O*NET matching can miss roles with unusual titles and include false hits; duplicates are collapsed only when company, title and description text are identical.
- 12 profiles and about 80 labels per reviewer is a demo go/no-go check, not proof that one method is generally better.
- The first embedding run ran out of memory at the merge step (job text was still held in RAM). Fixed: the merge now writes through a memory-mapped file and job text is released first (`finalize-encode`). No embeddings were regenerated.
- The job-text hash for the embedding run was not recorded (job text was not rebuilt on recovery); the vectors are tied to the corpus by job-ID order and the sample re-encode check.
- BGE query text for the real profiles contains resume-derived content; it ran locally on this machine only.

## Rules that still apply

- Missing information on a profile is **unknown**, not a gap. Listings are **historical**, not verified openings.
- Resumes and profile text containing personal details stay under the Git-ignored `data/` folder. Nothing is sent to an outside AI service unless the person whose resume it is agrees first. No provider calls were made for this direction.
- The six unopened weighted-RRF held-out profiles remain unopened.
- The shared OpenAI spend cap ($3, about $2.92 reserved) is effectively used up. Any paid call needs a new decision.

## Still pending

1. (Optional now) ~~Write target roles~~ Done 2026-10-09: both lists are recorded in `data/candidates/target_roles_TO_FILL_IN.md`, and the recall check was run on the saved rankings (`data/matching/ai_ml_rankings_20261009/recall_check_results.json`, patterns in `recall_target_patterns.json`). The method-by-method results are deliberately not looked at until you have both finished labelling, so the blind review is not biased. Still to do: label the review page and export labels.
2. Edit the draft "desired roles" on the two real profiles if they are wrong.
3. ~~Analyze labels~~ Dropped (no time). The demo ranker is RRF by default and is unvalidated.
4. ~~Connect the frontend to the backend~~ Done. Still to do: a short end-to-end rehearsal on your own profile, and a deployment or recording plan.
5. Live test of grounded explanations on your own resumes, only with your explicit agreement to send resume text to the provider and a new budget decision.
6. Clean notebook for the final path; demo recording.
7. Nice to have, decision pending (2026-10-09): per-user "Refine my matches" button trained on a user's own 👍/👎 ratings (plan, thresholds and risks in `roadmap.md`, section 6). Not started; revisit after the core demo and write-up.

## Run log

- 2026-10-09 11:40-12:16 UTC-5: full scan (`targeted_pool_20261009T164007Z`), complete.
- 12:17-12:19: corpus built (first build was 3.3 GB; rebuilt without the raw-HTML field, 2.0 GB, same 244,558 rows).
- 12:21-13:30: GPU embedding (RTX 3070, CUDA build of the same torch version installed under `data/_gpu_libs`, project venv unchanged): text build 1,553 s, encode 2,588 s. Merge step ran out of memory; all 60 chunks had been saved.
- 14:15-14:27 (after a restart): verified chunks, finished the merge with bounded memory, ranked 12 profiles (TF-IDF fit 359 s on 244,558 jobs), built the review page.
- 2026-10-09 evening: role-family classifier run (`role_family_classifier_20261009T205400Z`, 94.7% test accuracy, supporting model only). Backend restarted from the ML `.venv` (added fastapi, starlette, httpx there) with the RRF ranker on `ai_ml_serving_v1`; verified with a fictional profile.
