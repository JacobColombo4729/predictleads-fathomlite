# Asks

> What each of us wants the other to make sure of. Eve adds an ask when you answer its last question at push time, then turns it into a question on the other person's pushes until their work takes care of it (noted in their change log). Their answers appear as `↳` replies under the ask. Edit by hand any time.

## For Jesse (from Jacob)

- **Combining our models (2026-10-15).** Read `app/docs/combined-model-plan.md` and say whether you agree before I build it. ML_Fit's trained layer, which builds on your TF-IDF + BGE scores, would replace the stand-in fit factor in the `mlrank` queue. Your RRF ranker keeps picking the candidates, with your feedback, Compass and similar-to-liked kept. Two points need your call: widening the candidate pool to the top text matches within the person's top roles (`rrf_ranker.py`, step 4), and what drives the Compass and role suggestions (step 6).

- **Text blockers and title rules are labels, not hiding (2026-10-13).** Years asked, degree with no equivalent, clearance, no sponsorship, "marketing specialist" and "staffing agency, from the title" are unvalidated keyword rules: show them as listing labels and opt-in filters (each with a count of what it would hide), never hidden by default. Only seniority (manager or leadership for users without management experience), likely ghosts and document management specialists hide by default. See `app/DESIGN.md` and the Decisions line in `JACOB.md`.

- **Helper agents are optional (2026-10-13).** Eve, Thread, Pace and Foreman cost Claude usage, so none runs unless you ask for it by name; your Claude won't suggest them or turn Eve's hooks on. If you ever want Eve: `git config core.hooksPath .githooks` (README step 14). No action needed.

- **The model is built only from evidence (2026-10-13).** Training, tuning, evaluation and reported results use only the scope's listings and Profiles' public résumés, never our own résumés, so the model has no personal bias. Entering your résumé in the running app as a user is fine: the finished model just scores it. Delete your local `data/candidates/real_candidate_profiles.json` and any review files built from `real_jacob` / `real_jesse` profiles. See `app/APP.md` (Now).

## For Jacob (from Jesse)

_None open._
