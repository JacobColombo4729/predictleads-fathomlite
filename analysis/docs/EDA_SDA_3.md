<!-- Generated from notebooks/EDA_SDA_3.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: EDA/SDA 3 (Text features)

PredictLeads × TripleTen externship · Jacob Colombo

## 0. Purpose

**What do the listing descriptions say?** This notebook turns the description text of EDA_SDA_2's scope listings into **features**, and reports the text rules as **keyword rules with their firing rates**, so `EDA_SDA_4` can test every feature (text included) and the app can label what a listing asks and offer opt-in filters. **Decided 2026-10-13: the validation samples (section 5) will not be hand-checked,** so the text blockers (years asked, degree required, clearance, no sponsorship and the other career-change flags) are **unvalidated**. They are labels on each listing and filters a career changer turns on (for example "I have no degree"), never hidden by default. The same decision covers EDA_SDA_2's two title rules (staffing agency from recruiter titles; the 13-1161 split into analysts and marketing specialists): they are not hand-checked here either, and are labels and opt-in filters. Only reliable rules hide by default (seniority, likely ghosts, occupations that aren't a tech path); every rule reports how many listings it would hide. **It builds no matching models:** résumé-to-listing fit (keyword overlap, TF-IDF, BGE embeddings, RRF) is in `ML_Fit`, which uses this notebook's cleaned text and skills. A text feature is kept only if it adds information beyond the non-text features.

**For career changers,** the most important text features are what a listing asks of a newcomer (experience, education, licenses, training) and which of its skills carry over from other fields.

**Inputs:** EDA_SDA_2's scope table, plus the description text for those listings, loaded from the saved flattened parts (EDA_SDA_1's table keeps only description features, to fit in memory).

**The series** (each notebook hands its final table to the next):

| Notebook | Question | Hands off |
|:--|:--|:--|
| EDA_SDA_1 | What's in the full file, how does tech compare with every field, and what's in the scope? (the scope cut, applied first) | Scope table and its descriptions, occupation summary, company activity |
| EDA_SDA_2 | Which scope listings do we keep, and how do they behave? Additional cleaning, non-text checks by role bucket, ghost listings | Scope table, non-text features |
| EDA_SDA_3 | What do the descriptions say? Text features, unvalidated text rules as labels and opt-in filters, skills mapped to O*NET (no matching models) | Scope table + text features |
| EDA_SDA_4 | Which features hold up? Statistics, confounders, the adjusted Cox check | Modeling-ready table |
| TimeSeries | When do listings appear and close? | Timing features |
| ML_Fit | Which roles and listings fit a career changer? Keyword → TF-IDF → BGE → RRF, skills-based matching, a trained layer on real career moves | Fit scores, role discovery, the compass |
| ML_Rank | How should listings be ranked? The listing timing model, combined with fit into the priority queue | The priority queue algorithm |

**Standard:** the TripleTen course templates are the baseline (structure, methods, reporting). Every applicable course method is used, and taken further with more advanced techniques where the data calls for them.

**View:** listings with a description (82% of the scope; EDA_SDA_2 3.9). Text features are **unknown** for the rest, never "no", and "has a description" stays a standard confounder: listings without one close in about 3 days.

**Data rules:** aggregates only; never print descriptions, titles or sample rows, except in cells tagged `data-rows` (review cells for a teammate to check locally; cleared before every commit). Text stays local: no listing text goes to outside AI services. Clear outputs before committing.

### Clarifying questions
- What does description length actually capture: employer intent, or boilerplate?
- Does urgency wording, read in context rather than by keyword, relate to how long listings stay open? (EDA_SDA_1 left it here: its keyword flag misfired and showed no gap.)
- Can evergreen wording ("always accepting applications") identify ghost listings?
- Are EDA_SDA_1's career-change flags (years required, degree required, training and career-changer wording) accurate, and what do they miss? Two are urgent: "asks 5+ years" would hide 35% of open listings if a user turned it on, and the degree flag fires on only 3.7% (EDA_SDA_2 3.11).
- Which requirements are labels and opt-in filters (years asked, degree, clearance, sponsorship, certification), and how many listings would each hide if a user turned it on? (Not validated: see the 2026-10-13 decision above.)
- Which skills does each listing ask for, mapped to O*NET skills, so `ML_Fit` can match on skills rather than words?

### Roadmap

1. **Setup:** the scope table and its description text
2. **Description cleanup:** HTML to structured text, boilerplate removal, normalization and lemmas
3. **The text rules:** years of experience (required vs. preferred) and degree requirements, kept as labels and opt-in filters
4. **Other text features:** evergreen wording, hiring-difficulty phrases, licenses and certifications, training and career-changer wording, urgency, remote, length
5. **Validation samples:** written, but not being marked (decided 2026-10-13); rules are reported as keyword rules with their firing rates
6. **Skills mapped to O*NET** (for `ML_Fit`)
7. **Final table:** text rules saved as labels with the count each would hide (nothing hidden by default, nothing removed), for EDA_SDA_4 and `ML_Fit`
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
| Education required | "Bachelor's required" vs. "or equivalent experience"; field-specific degrees | EDA_SDA_1's pattern only fires when "required" comes within six words of "bachelor's", so the most common wording ("Bachelor's degree in Computer Science or a related field") never matches: it flags only 3.7% of listings. Rebuilt in section 3, with requirements and preferred sections told apart |
| **Clearance, sponsorship, travel, shifts** (added 2026-10-10) | Security clearance required; "no visa sponsorship"; travel percentage; shift or on-call work; contract-to-hire | Clearance and sponsorship are blockers for some career changers: a label on the listing and an opt-in filter in the app, like a missing degree (never hidden by default) |
| Licenses and certifications required | "Security+", "CCNA", "CISSP" and the like | A required credential the career changer lacks: a label and an opt-in filter |
| Training and career-changer wording | "Will train", "paid training", "career changers welcome", "transferable skills" | Reports EDA_SDA_1's flags as keyword rules with their firing rates (the hand-check sample is not being marked, decided 2026-10-13) |
| **Text-rule filters** | Which years-asked threshold and which required credentials would a user filter on, per role bucket? | Not validated (decided 2026-10-13): saved in this notebook's final table as labels, with a count of the listings each would hide if turned on (opt-in in the app, never hidden by default, never removed) |
| Required vs. preferred qualifications | Split on headings like "Requirements" and "Nice to have" | Part of fit and gap analysis in `ML_Fit` |
| **Skills, mapped to O*NET** | Extracted with spaCy (PredictLeads `tags` are sparse) and mapped to O*NET skills | `ML_Fit` matches résumés to listings on these skills, so "transferable" doesn't depend on the old field's vocabulary |

**Moved to `ML_Fit`:** role similarity, TF-IDF and embeddings for matching. **Moved to `ML_Rank`:** TF-IDF terms that predict how fast a listing closes (a text input to the timing model). Topic modeling stays a stretch goal.

## Plan: preprocessing

Course standard (MLforTexts): normalize text (lowercase, regex to keep letters), remove NLTK stopwords, lemmatize with spaCy. Advanced: remove boilerplate where headings make it detectable. Any vocabulary learned from the text is fit on the training months only.

**Evaluation (course standard):** the plan was precision and recall on a hand-checked random sample against EDA_SDA_1's keyword rule. **Decided 2026-10-13: the samples will not be hand-checked,** so each rule is reported as a keyword rule with its firing rate, against EDA_SDA_1's rule where one exists, and not as a validated blocker.

## 2. Description cleanup

**Why:** every text feature, and `ML_Fit`'s matching, should read what the listing asks for, not the legal and benefits text most listings share. At full scale the descriptions are several GB of text, so **the cleanup runs batch by batch** (`src/text_features.py`): each batch is cleaned, its features computed and both written to disk, then dropped from memory.

| Step | What | Output |
|:--|:--|:--|
| 2a | HTML to text, one line per paragraph, list item or heading (long unbroken text split into sentences) | Lines per listing |
| 2b | Remove boilerplate line by line: known patterns (equal opportunity and legal, benefits, application logistics), "About the company" sections, and text shared word for word across 5+ employers (learned on a sample of April–July listings; lines with a requirement cue are never cut) | `desc_clean` (`eda3_desc_clean.parquet`), `words_clean` |
| 2c | Course standard: lowercase, letters only, NLTK stopwords, spaCy lemmas, on a stratified sample | `desc_lemmas` (for word counts and TF-IDF later) |

The rules in sections 3–4 run on the cleaned text, which keeps case, numbers and punctuation ("3+ years", "B.S.", "Security+"), and tell **requirement lines from preferred ones** using the headings kept in 2a.

**What the data shows**
- **The cleanup ran on all 908,291 scope listings, 82.4% of which have a description,** in 15.6 minutes (batch by batch). The shared template lines were learned on 120,000 April–July listings and found 6,556.
- **Boilerplate removed, by kind (words, share of all description words):** equal opportunity and legal 36.1M (6.6%), about the company 24.1M (4.4%), benefits 23.9M (4.4%), shared template text 20.4M (3.7%), application logistics 2.5M (0.40%).
- **Listings keep most of their text:** the median listing keeps 85–86% of its words in every source type (company site 716 to 559 words, hiring platform 784 to 617, job board 620 to 496), and none is cleaned to nothing.

**What the data shows**
- **The chart:** the word counts are two humps on the log scale, a small group of very short descriptions (under about 60 words) and the main body at about 300-1,200 words; boilerplate removal shifts the main body left without changing its shape, and the lemma step moves it to about 250-500 words, leaving a spike of near-empty sampled listings at 1 word. On the right, equal opportunity and legal text is the largest block removed (6.6% of all words), then about the company and benefits (4.4% each), shared template text (3.7%) and application logistics (0.4%).
- **Words per description fall at each step:** a median of 735 after HTML removal, 573 after boilerplate removal and 394 as lemmas with stopwords removed (the lemma step is on a 150,001-listing sample). Means are 732, 590 and 406.
- **The longest descriptions are long before and after:** the 90th percentile is 1,149 words after HTML removal and 957 after boilerplate removal, and the maximum (7,827 words) is the same at both steps.

## 3. The blocker rules: years of experience and degree

EDA_SDA_1's rules read every line the same way: "3+ years required" and "5 years preferred" both count, and its degree rule only fires when "required" comes within six words of "bachelor's", so the most common wording ("Bachelor's degree in Computer Science or a related field") never matched (3.7% of listings). Here both are rebuilt on the cleaned text, with **requirement lines told apart from preferred ones** (from the headings kept in 2a, and lines that say "preferred" or "a plus"):

- **Years:** `years_asked` (any line, like EDA_SDA_1), `years_required` (requirement lines only), `years_preferred`. Also reads "minimum of two years" and number words.
- **Degree:** `degree_required` (a degree named on a requirement line), `degree_preferred`, `degree_equivalent` ("or equivalent experience", "in lieu of a degree"), `degree_level` (the highest level required).

The question for the app: **how many listings would each rule hide if a user turned it on** (too many years required; a degree they don't have). These are unvalidated keyword rules (decided 2026-10-13), so they label listings and drive opt-in filters; nothing hides by default.

**What the data shows**
- **The chart:** the requirement-line rule finds 5+ years in more listings than EDA_SDA_1's rule in every bucket but business and data analysis (about 24% vs. 26%), most in security (about 39% to 43%), networks and systems, other computer roles and quality assurance; the order barely changes, from IT project management (about 49%) at the top to IT support (about 14%) at the bottom.
- **EDA_SDA_1's years rule was too narrow:** it finds 5+ years in 33.7% of listings with a description, the rebuilt rule in 39.1% (any line). The new rule finds more years in 12% of listings and fewer in 1.2%; both say no years in 34%.
- **Required vs. preferred matters a little for years:** 5+ years on a requirement line is 35.6% of listings (36% of those open on September 30), and 3.4% ask it only on preferred lines. A filter on 5+ years would hide about a third of open listings if a user turned it on, so the threshold has to be chosen per role bucket, not once.
- **The years asked climb with the bucket:** the median required is 3 years in IT support and marketing specialists, 4 in business and data analysis, 5 in most buckets and 6 in IT project management; 5+ required runs from 14% (IT support) to 49% (IT project management), and 2 years or fewer from 23% (IT support) down to 5.1% (IT project management).
- **EDA_SDA_1's degree rule fired on 3.7% of listings; the rebuilt rule finds a degree on a requirement line in 52%,** another 11% only on preferred lines, and 63% anywhere. 29% accept "or equivalent experience", and 29% require a degree with no equivalent (the "I have no degree" filter).
- **Most requirements are a bachelor's:** of the highest level named on a requirement line, 34% are bachelor, 9.5% master, 3.6% PhD and 0.90% associate; 52% name none.
- **The degree blocker is lowest in IT support (20% with no equivalent) and highest in quality assurance (37%),** with business and data analysis and data science at 36%.
- **These are unvalidated keyword rules:** the section 5 samples will not be marked (decided 2026-10-13), so these rates are firing rates, not precision. They label listings and drive opt-in filters; nothing hides by default.

## 4. Other text features

Each feature is a keyword rule on the cleaned text (`src/text_features.py`), reported with its firing rate and not hand-validated (decided 2026-10-13); it labels listings and can be an opt-in filter, and hides nothing by default. For each: how often it fires, and whether it relates to how fast listings close **beyond the confounders** (the 7-day close rate's gap with vs. without the feature, within company volume band × source type cells, on original listings with a full week of follow-up; EDA_SDA_4 tests the survivors properly). Listings where a feature is unknown (NA) are left out of that feature's comparison, not counted as "without"; the `known` column in 4b shows each feature's coverage (1 = every listing in the comparison is known).

| Group | Features |
|:--|:--|
| Labels and opt-in filters for some applicants | `clearance_required` (vs. `clearance_obtainable`), `no_sponsorship`, `us_citizen_required`, `cert_required` and named certifications, `shift_or_oncall`, `travel_pct`, `contract_to_hire` |
| What a newcomer is offered | `entry_wording`, `changer_wording` (now also bootcamp graduates, self-taught) |
| Hiring difficulty and urgency | `sign_on_bonus`, `hiring_immediately`, `relocation_offered`, `urgency_context` ("urgent care" no longer counts) |
| Evergreen wording (ghost signal) | six phrase groups and `evergreen_any`, compared with EDA_SDA_2's likely-ghost flag |
| Remote and length | `remote_mention` (minus "remote patient monitoring" and the like), `words_clean` |

**What the data shows**
- **Most features are rare.** Remote is mentioned in 29%, a certification in 11%, a required clearance in 9.8% and no sponsorship in 6.9%; shift or on-call work 5.2%, US citizenship required 4.8%, entry-level wording 3.8%. Career-changer wording is on only 0.45% of listings, and a required certification on 0.91%.
- **Role buckets differ where you would expect:** certifications are mentioned in 40% of security listings (CompTIA 17%, security certifications 33%), 22% of IT support (CompTIA 17%) and 20% of IT project management (project certifications 15%); clearance is required in 26% of security, 15% of quality assurance and 14% of networks and systems listings, but 1.3% of technical sales. Shift or on-call work is 13% of IT support.
- **The 7-day close-rate gaps are modest.** About 21% of the original listings close within a week. Within company volume band × source type cells, the gaps run from -5.6 points (always-hiring wording, 2.1% of listings) to +5.4 points (urgency in context, 0.20%). Next are sign-on bonus (-3.9), any evergreen wording (-3.0), a required degree (-2.8) and Cisco certifications (-2.8). Features on 0.5% of listings or fewer are noisy.
- **Some raw gaps change sign once the confounders are held fixed:** contract-to-hire goes from +4.3 to -1.0 points and no-opening wording from -4.6 to +2.9. This is a first look, not a causal estimate; EDA_SDA_4's adjusted model decides which survive. In this run no listing with a description has an unknown feature, so the `known` share is 1 for every row.
- **Evergreen wording does not line up with the ghost flag:** any evergreen phrase is on 4.0% of closed listings, 3.3% of likely ghosts, 4.5% of listings open 90+ days from companies still posting and 4.2% of open listings 90 days or younger. As it stands it does not add to "old and from a quiet company"; always-hiring wording (2.1% of listings) is the largest of the six groups.

## 5. Validation samples (not being marked)

**Decided 2026-10-13: these samples will not be hand-checked.** 5a still writes a **sample** for each rule: 60 listings where it fires and 60 where it doesn't, stratified by role bucket, to `data/clean/eda3_validation/<rule>.csv` (local only), and 5b would read marks back and report precision and recall if anyone ever fills them in. Nobody is marking them, so **the rules are reported as keyword rules with their firing rates (sections 3, 4 and 7), not as validated blockers.** That sets what each rule may do: the text blockers (years asked, degree required with no equivalent, clearance, no sponsorship, required certification and the other career-change flags) are **labels on each listing and opt-in filters a career changer turns on, never hidden by default**. Only reliable rules hide by default (seniority, likely ghosts, occupations that aren't a tech path, all decided in EDA_SDA_2). Every rule reports how many listings it would hide (7a). Listing text never leaves the laptop.

**What the data shows**
- **No rule is validated, and none will be:** the samples are written (about 120 rows per rule, for ten rules: 5+ years required, degree required, degree equivalent, clearance, no sponsorship, required certification, entry wording, career-changer wording, evergreen wording, urgency) but are not being marked (decided 2026-10-13), so 5b reports no precision or recall.
- **So the rules are keyword rules reported with their firing rates:** they are labels and opt-in filters, never hidden by default; the 90% precision bar that would have let a rule hide listings is not applied to any of them.

## 6. Skills mapped to O*NET

**Waiting for the O*NET download** (terms of use first; `analysis/ANALYSIS.md`). Then: skills extracted from the requirement lines (spaCy noun phrases plus a skills vocabulary), mapped to O*NET skills for `ML_Fit`'s skills-based matching. The cleaned text it needs is already saved (`eda3_desc_clean.parquet`).

## 7. Final table

One row per scope listing (`id`), the text features from sections 3–4, and **text-rule flags** for the app (the `blocker_*` columns). They are unvalidated keyword rules (decided 2026-10-13): the app shows them as labels and opt-in filters (for example "I have no degree"), never hides by default, and 7a reports how many listings each would hide. Blank means unknown: the listing has no description. Saved as `text_table.parquet` for EDA_SDA_4 (joined by `id`) and `ML_Fit`.

## 8. Conclusion

**Summary**
- **Cleanup:** 908,291 scope listings, 82.4% with a description; the median description falls from 735 words to 573 after boilerplate removal (85–86% of text kept per source type). Equal opportunity and legal text is the largest share removed (6.6% of all words).
- **Rebuilt rules (unvalidated keyword rules):** 5+ years on a requirement line is 35.6% of listings (36% of open ones), against 33.7% for EDA_SDA_1's rule; a degree is required on 52% (EDA_SDA_1: 3.7%), 29% with no equivalent accepted.
- **Other features:** most are rare (remote 29%, certification 11%, clearance 9.8%, no sponsorship 6.9%, career-changer wording 0.45%). Their 7-day close gaps within confounders stay within about 5.6 points either way.
- **Evergreen wording** does not concentrate in likely-ghost listings (3.3%, against 4.0–4.5% elsewhere).

**Answers to the clarifying questions**
- *Length: intent or boilerplate?* Not answered here; the cleanup shows boilerplate is a large part of the text, and length (`words_clean`) is saved for EDA_SDA_4.
- *Urgency in context vs. close time?* It fires on 0.20% of listings, with a +5.4-point gap within confounders: too few listings to rely on.
- *Evergreen wording as a ghost signal?* Not as it stands (see above).
- *Are EDA_SDA_1's flags accurate?* The years rule misses some (the new rule finds more in 12% of listings) and the degree rule missed most requirements (3.7% against 52%). Precision is untested: the hand-check samples will not be marked (decided 2026-10-13).
- *Which requirements hide a listing?* None by default. Years asked, clearance, no sponsorship, required certification and a degree with no equivalent are **labels and opt-in filters**, and would hide this much if a user turned each on: 5+ years required (35.6% of listings), clearance (9.8%), no sponsorship (6.9%), a required certification (0.91%) and a degree with no equivalent (29%). **None is validated (the section 5 samples will not be marked, decided 2026-10-13), so none hides by default.**
- *Skills mapped to O\*NET?* Pending the O\*NET download (section 6).

**Limits:** the gaps in 4b are a first look and associational, not causal; the rules are unvalidated keyword rules (firing rates, no precision); the O\*NET skills and the length question are open.

**Handoff:** `text_table.parquet` (908,291 listings, 45 columns, text features and text-rule flags (labels, unvalidated), joined by `id`; blank where there is no description) for EDA_SDA_4 and `ML_Fit`; `eda3_desc_clean.parquet` (cleaned text) and `eda3_lemmas_sample.parquet` (150,001 listings) for `ML_Fit`.
