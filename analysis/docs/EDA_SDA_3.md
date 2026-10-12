<!-- Generated from notebooks/EDA_SDA_3.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: EDA/SDA 3 (Text features)

PredictLeads × TripleTen externship · Jacob Colombo

## 0. Purpose

**What do the listing descriptions say?** This notebook turns the description text of EDA_SDA_2's scope listings into **features**, and validates the text rules that hide listings, so `EDA_SDA_4` can test every feature (text included) and the app can hide what a career changer can't get. **It builds no matching models:** résumé-to-listing fit (keyword overlap, TF-IDF, BGE embeddings, RRF) is in `ML_Fit`, which uses this notebook's cleaned text and skills. A text feature is kept only if it adds information beyond the non-text features.

**For career changers,** the most important text features are what a listing asks of a newcomer (experience, education, licenses, training) and which of its skills carry over from other fields.

**Inputs:** EDA_SDA_2's scope table, plus the description text for those listings, loaded from the saved flattened parts (EDA_SDA_1's table keeps only description features, to fit in memory).

**The series** (each notebook hands its final table to the next):

| Notebook | Question | Hands off |
|:--|:--|:--|
| EDA_SDA_1 | What's in the full file, how does tech compare with every field, and what's in the scope? (the scope cut, applied first) | Scope table and its descriptions, occupation summary, company activity |
| EDA_SDA_2 | Which scope listings do we keep, and how do they behave? Additional cleaning, non-text checks by role bucket, ghost listings | Scope table, non-text features |
| EDA_SDA_3 | What do the descriptions say? Text features, validated blockers, skills mapped to O*NET (no matching models) | Scope table + text features |
| EDA_SDA_4 | Which features hold up? Statistics, confounders, the adjusted Cox check | Modeling-ready table |
| TimeSeries | When do listings appear and close? | Timing features |
| ML_Fit | Which roles and listings fit a career changer? Keyword → TF-IDF → BGE → RRF, skills-based matching, a trained layer on real career moves | Fit scores, role discovery, the compass |
| ML_Rank | How should listings be ranked? The listing timing model, combined with fit into the priority queue | The priority queue algorithm |

**Standard:** the TripleTen course templates are the baseline (structure, methods, reporting). Every applicable course method is used, and taken further with more advanced techniques where the data calls for them.

**View:** listings with a description (77% of the scope; EDA_SDA_2 3.9). Text features are **unknown** for the rest, never "no", and "has a description" stays a standard confounder: listings without one close in about 3 days.

**Data rules:** aggregates only; never print descriptions, titles or sample rows, except in cells tagged `data-rows` (review cells for a teammate to check locally; cleared before every commit). Text stays local: no listing text goes to outside AI services. Clear outputs before committing.

### Clarifying questions
- What does description length actually capture: employer intent, or boilerplate?
- Does urgency wording, read in context rather than by keyword, relate to how long listings stay open? (EDA_SDA_1 left it here: its keyword flag misfired and showed no gap.)
- Can evergreen wording ("always accepting applications") identify ghost listings?
- Are EDA_SDA_1's career-change flags (years required, degree required, training and career-changer wording) accurate, and what do they miss? Two are urgent: "asks 5+ years" would hide 31% of open listings, and the degree flag fires on only 3–6% (EDA_SDA_2 3.11).
- Which requirements are blockers for everyone (hide the listing), and which depend on the applicant (degree, certification)?
- Which skills does each listing ask for, mapped to O*NET skills, so `ML_Fit` can match on skills rather than words?

### Roadmap

1. **Setup:** the scope table and its description text
2. **Description cleanup:** HTML to structured text, boilerplate removal, normalization and lemmas
3. **The blocker rules:** years of experience (required vs. preferred) and degree requirements
4. **Other text features:** evergreen wording, hiring-difficulty phrases, licenses and certifications, training and career-changer wording, urgency, remote, length
5. **Validation:** precision and recall of each rule on hand-checked samples
6. **Skills mapped to O*NET** (for `ML_Fit`)
7. **Final table:** validated blockers applied (hidden, never removed), saved for EDA_SDA_4 and `ML_Fit`
8. **Conclusion**

## Plan: text features to build

| Feature | Question | Notes |
|:--|:--|:--|
| **Description cleanup** (moved here from EDA_SDA_2 2b) | Strip HTML and boilerplate (equal-opportunity statements, benefits, "about us") | `desc_clean` and `desc_words_clean`; the base for every feature below and for `ML_Fit` |
| Description length | Is it employer intent, or just boilerplate? | Measured on `desc_clean` |
| Urgency wording | Does real urgency (in context, not "urgent care") go with shorter or longer windows? | EDA_SDA_1's keyword flag showed no gap; may mark hard-to-fill roles |
| Remote mentions | Does remote add beyond `contract_types`? | |
| **Evergreen wording** | Listings that collect applications instead of filling one job: a ghost-listing signal | Six phrase groups, each scored and checked on its own: **talent pool** ("join our talent community", "future opportunities"), **no specific opening** ("general application", "not for a specific position"), **always hiring** ("always accepting applications", "rolling basis"), **many openings in one ad** ("multiple openings", "various locations"), **pipeline labels** ("pipeline requisition", "evergreen requisition"), **staffing-style** ("on behalf of our client"); plus a generic-title check ("Software Engineer – All Levels"). Compared with EDA_SDA_2's likely-ghost flag (3.1): kept only if it adds to "old and from a quiet company" |
| Hiring-difficulty phrases | "Sign-on bonus", "hiring immediately", "relocation assistance", "will train" | Pairs with reposts and listing age |
| Years of experience required | "3+ years": is EDA_SDA_1's rule accurate? Required or preferred? In the field, or any experience? | The most common blocker for career changers |
| Education required | "Bachelor's required" vs. "or equivalent experience"; field-specific degrees | EDA_SDA_1's pattern only fires when "required" comes within six words of "bachelor's", so the most common wording ("Bachelor's degree in Computer Science or a related field") never matches: it flags only 3–6% of listings. Rebuilt in section 3, with requirements and preferred sections told apart |
| **Clearance, sponsorship, travel, shifts** (added 2026-10-10) | Security clearance required; "no visa sponsorship"; travel percentage; shift or on-call work; contract-to-hire | Clearance and sponsorship are hard blockers for some career changers: hidden per applicant in the app, like a missing degree |
| Licenses and certifications required | "Security+", "CCNA", "CISSP" and the like | A required credential the career changer lacks is a blocker |
| Training and career-changer wording | "Will train", "paid training", "career changers welcome", "transferable skills" | Validates EDA_SDA_1's flags: a team member checks a random sample of matches locally (never shared with AI tools) |
| **Hard-blocker rules** | Which years-asked threshold and which required credentials hide a listing from everyone, per role bucket? | Validated here, then applied in this notebook's final table, with a count of the listings each rule hides (hidden in the app, never removed) |
| Required vs. preferred qualifications | Split on headings like "Requirements" and "Nice to have" | Part of fit and gap analysis in `ML_Fit` |
| **Skills, mapped to O*NET** | Extracted with spaCy (PredictLeads `tags` are sparse) and mapped to O*NET skills | `ML_Fit` matches résumés to listings on these skills, so "transferable" doesn't depend on the old field's vocabulary |

**Moved to `ML_Fit`:** role similarity, TF-IDF and embeddings for matching. **Moved to `ML_Rank`:** TF-IDF terms that predict how fast a listing closes (a text input to the timing model). Topic modeling stays a stretch goal.

## Plan: preprocessing

Course standard (MLforTexts): normalize text (lowercase, regex to keep letters), remove NLTK stopwords, lemmatize with spaCy. Advanced: remove boilerplate where headings make it detectable. Any vocabulary learned from the text is fit on the training months only.

**Evaluation (course standard):** each validated rule reports precision and recall on a hand-checked random sample (checked locally by a teammate), against the keyword rule from EDA_SDA_1 as the baseline.

## 2. Description cleanup

**Why:** every text feature, and `ML_Fit`'s matching, should read what the listing asks for, not the legal and benefits text most listings share. At full scale the descriptions are several GB of text, so **the cleanup runs batch by batch** (`src/text_features.py`): each batch is cleaned, its features computed and both written to disk, then dropped from memory.

| Step | What | Output |
|:--|:--|:--|
| 2a | HTML to text, one line per paragraph, list item or heading (long unbroken text split into sentences) | Lines per listing |
| 2b | Remove boilerplate line by line: known patterns (equal opportunity and legal, benefits, application logistics), "About the company" sections, and text shared word for word across 5+ employers (learned on a sample of April–July listings; lines with a requirement cue are never cut) | `desc_clean` (`eda3_desc_clean.parquet`), `words_clean` |
| 2c | Course standard: lowercase, letters only, NLTK stopwords, spaCy lemmas, on a stratified sample | `desc_lemmas` (for word counts and TF-IDF later) |

The rules in sections 3–4 run on the cleaned text, which keeps case, numbers and punctuation ("3+ years", "B.S.", "Security+"), and tell **requirement lines from preferred ones** using the headings kept in 2a.

**What the data shows**
- (fill in after the run)

**What the data shows**
- (fill in after the run)

## 3. The blocker rules: years of experience and degree

EDA_SDA_1's rules read every line the same way: "3+ years required" and "5 years preferred" both count, and its degree rule only fires when "required" comes within six words of "bachelor's", so the most common wording ("Bachelor's degree in Computer Science or a related field") never matched (3–6% of listings). Here both are rebuilt on the cleaned text, with **requirement lines told apart from preferred ones** (from the headings kept in 2a, and lines that say "preferred" or "a plus"):

- **Years:** `years_asked` (any line, like EDA_SDA_1), `years_required` (requirement lines only), `years_preferred`. Also reads "minimum of two years" and number words.
- **Degree:** `degree_required` (a degree named on a requirement line), `degree_preferred`, `degree_equivalent` ("or equivalent experience", "in lieu of a degree"), `degree_level` (the highest level required).

The question for the app: **which rule can hide a listing from everyone** (too many years required), and which hide it only per applicant (a degree they don't have). Nothing hides until section 5 validates the rule.

**What the data shows**
- (fill in after the run)

## 4. Other text features

Each feature is a keyword rule on the cleaned text (`src/text_features.py`), validated in section 5 before it's used to hide anything. For each: how often it fires, and whether it relates to how fast listings close **beyond the confounders** (the 7-day close rate's gap with vs. without the feature, within company volume band × source type cells, on original listings with a full week of follow-up; EDA_SDA_4 tests the survivors properly).

| Group | Features |
|:--|:--|
| Blockers for some applicants | `clearance_required` (vs. `clearance_obtainable`), `no_sponsorship`, `us_citizen_required`, `cert_required` and named certifications, `shift_or_oncall`, `travel_pct`, `contract_to_hire` |
| What a newcomer is offered | `entry_wording`, `changer_wording` (now also bootcamp graduates, self-taught) |
| Hiring difficulty and urgency | `sign_on_bonus`, `hiring_immediately`, `relocation_offered`, `urgency_context` ("urgent care" no longer counts) |
| Evergreen wording (ghost signal) | six phrase groups and `evergreen_any`, compared with EDA_SDA_2's likely-ghost flag |
| Remote and length | `remote_mention` (minus "remote patient monitoring" and the like), `words_clean` |

**What the data shows**
- (fill in after the run)

## 5. Validation

Each rule that could hide a listing, or that feeds the model, gets a **hand-checked sample**: 60 listings where it fires and 60 where it doesn't, stratified by role bucket, written to `data/clean/eda3_validation/<rule>.csv` (local only) with the line that triggered it. A teammate marks each row `correct` = 1 or 0 locally; the notebook reads the marks back and reports **precision** (of the matches, how many are right) and **recall** (of the misses, how many should have matched), against EDA_SDA_1's rule where one exists. **A rule hides listings only if its precision is 90%+.** Listing text never leaves the laptop.

**What the data shows**
- (fill in after the run)

## 6. Skills mapped to O*NET

**Waiting for the O*NET download** (terms of use first; `analysis/ANALYSIS.md`). Then: skills extracted from the requirement lines (spaCy noun phrases plus a skills vocabulary), mapped to O*NET skills for `ML_Fit`'s skills-based matching. The cleaned text it needs is already saved (`eda3_desc_clean.parquet`).

## 7. Final table

One row per scope listing (`id`), the text features from sections 3–4, and **candidate blocker flags** for the app. A flag only hides listings once its rule passes validation (section 5); until then it's saved, not applied. Blank means unknown: the listing has no description. Saved as `text_table.parquet` for EDA_SDA_4 (joined by `id`) and `ML_Fit`.

## 8. Conclusion

- (fill in after the run: the cleanup, the rebuilt years and degree rules, the other text features, what validation shows, and what goes to EDA_SDA_4 and ML_Fit)

**Handoff:** `text_table.parquet` (text features and candidate blockers, joined by `id`) for EDA_SDA_4 and `ML_Fit`; `eda3_desc_clean.parquet` (cleaned text) and `eda3_lemmas_sample.parquet` for `ML_Fit`.
