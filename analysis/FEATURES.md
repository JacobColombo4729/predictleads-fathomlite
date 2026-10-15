# Features by bucket

The features the project uses to match and prioritize listings, grouped by what they measure. Sources: the prioritization factors in `PROJECT_OUTLINE.md` and EDA_SDA_1's Q1 table; what survived is decided in EDA_SDA_4 (`modeling_ready.parquet`).

**What the queue uses:** ML_Rank's priority queue scores four factors (**fit, still open, accessibility, pay**; ML_Rank 8d). Buckets 5–7 feed the "still open" listing model rather than being scored on their own.

## 1. Fit: does the person suit the role?
- **Transferable fit:** résumé vs. role text (keyword, TF-IDF, BGE, RRF), plus O*NET skills linking the old and new occupations (ML_Fit 4–5)
- **ML_Fit's trained layer** on real career moves (Profiles)
- **Role bucket** (the 8 roles) and **O*NET occupation group**

## 2. Timing: how fast to move
- **Still open / apply-by (the due date):** the model's target, not an input; survival analysis on first- and last-seen dates
- **Freshness:** days since first seen

## 3. Accessibility: can a career changer get it?
- **Hides the listing:** seniority (manager level, for users without management experience)
- **Labels and opt-in filters:** years required, degree required, "or equivalent experience"
- **Raises it:** entry seniority, a low experience ask, entry wording, career-changer wording

## 4. Pay
- **Standardized hourly pay** (EDA_SDA_1 6.5) and whether pay is disclosed

## 5. Employer intent (listing text)
- **Evergreen wording:** "always accepting applications", six phrase groups plus `evergreen_any`
- **Urgency in context**, remote mentions, description length
- **Hiring-difficulty phrases:** sign-on bonus, hiring immediately, relocation offered

## 6. Competition and hiring difficulty
- **Company hiring volume** (`company_earlier`)
- **Reposts:** `is_repost`, `times_listed`, days since first and last listed
- **Listing age:** likely ghosts hidden; listings open 90+ days labelled and ranked lower

## 7. Source and company
- **Source type** and platform; whether the listing is on the employer's own domain
- **Company type:** publicly traded, has a parent company
- **Recruiters:** anonymous key and title

## 8. Growth
- **Expected 5-year job growth** for the role: a visible, adjustable weight in role discovery and the compass

## What didn't hold up
- **Evergreen wording** doesn't line up with the likely-ghost flag (EDA_SDA_3 section 4).
- **Urgency from cross-posting** had little support (EDA_SDA_2 3.5c), so it isn't a factor; `copy_type` and the repost features are kept.
