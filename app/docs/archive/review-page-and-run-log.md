# Review Page and Run Log (2026-10-09 to 2026-10-10)

> **Archive.** Moved here from `app/STATUS.md` on 2026-10-12. The human review was dropped for time; the page is still there if you want evidence later. Current state: [`STATUS.md`](../../../analysis/STATUS.md). Paths are relative to `app/` and point into the git-ignored `data/` folder.

## The human review page (optional)

Open `data/matching/ai_ml_rankings_20261009/review/review.html` (double-click it, or open it in a browser from the file path; nothing is uploaded).

- Pick your name (Jesse or Jacob) in the top bar. It shows only your assigned pairs (82 each; 139 pairs in total, 25 labelled by both of you for an agreement check). Use the same way of opening the file every time: labels are saved in that browser, so press **Export my labels** when you stop and keep the file.
- Method, rank and score are hidden and the order is shuffled. The answer key is `.../review/answer_key.json`: do not open it until you have both finished.
- After you both export (`labels_jesse.json`, `labels_jacob.json`): `python code/analyze_review_labels.py --run-dir data/matching/ai_ml_rankings_20261009 --labels labels_jesse.json labels_jacob.json`.
- Before you look at any rankings, fill in `data/candidates/target_roles_TO_FILL_IN.md`; then `python code/recall_check.py --run-dir data/matching/ai_ml_rankings_20261009 --profile-id <id> --titles "<role>" ...`. Done 2026-10-09: both lists are recorded, and the recall check was run on the saved rankings (`recall_check_results.json`, patterns in `recall_target_patterns.json`). The method-by-method results are deliberately not looked at until labelling is finished, so the blind review is not biased.
- Measuring the ranker this way: about 40 pairs each (about 30 minutes; 60-80 for a stronger result) gives the one real number for the pitch.

## Run log

- 2026-10-09 11:40-12:16 UTC-5: full scan (`targeted_pool_20261009T164007Z`), complete.
- 12:17-12:19: corpus built (first build was 3.3 GB; rebuilt without the raw-HTML field, 2.0 GB, same 244,558 rows).
- 12:21-13:30: GPU embedding (RTX 3070, CUDA build of the same torch version installed under `data/_gpu_libs`, project venv unchanged): text build 1,553 s, encode 2,588 s. Merge step ran out of memory; all 60 chunks had been saved.
- 14:15-14:27 (after a restart): verified chunks, finished the merge with bounded memory, ranked 12 profiles (TF-IDF fit 359 s on 244,558 jobs), built the review page.
- 2026-10-09 evening: role-family classifier run (`role_family_classifier_20261009T205400Z`, 94.7% test accuracy, supporting model only). Backend restarted from the ML `.venv` (added fastapi, starlette, httpx there) with the RRF ranker on `ai_ml_serving_v1`; verified with a fictional profile.
- 2026-10-10: Signals layer, Compass stretch slider, AI skills gap analysis, feedback log, reasons on thumbs-down and exploration built and pushed to `main` (journal entry 16). Start the backend with `.\run_demo_backend.ps1 -ResumeFill -Assess`; the first live gap-analysis run failed on next-step rules and was fixed the same day.

## Budget

The shared OpenAI spend cap ($3, about $2.92 reserved) was effectively used up; an extra $2 was approved 2026-10-09, of which a few cents were spent on live tests. Any paid call needs a new decision.
