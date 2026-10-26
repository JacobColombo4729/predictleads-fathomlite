<!-- Generated from notebooks/EDA_SDA_3_Lean.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# EDA/SDA 3 (Lean): Text Features

PredictLeads × TripleTen externship · Jacob Colombo

## Purpose

**What do the listing descriptions say?** This notebook turns the description text of the Lean chain's in-scope listings into **features**, and reports the text rules as **keyword rules with their firing rates**, so `EDA_SDA_4` can test every feature (text included) and the app can label what a listing asks and offer opt-in filters. **Decided 2026-10-13: the validation samples (section 5) will not be hand-checked,** so the text blockers (years asked, degree required, clearance, no sponsorship and the other career-change flags) are **unvalidated**. They are labels on each listing and filters a career changer turns on (for example "I have no degree"), never hidden by default. The same decision covers EDA_SDA_2_Lean's two title rules (staffing agency from recruiter titles; the 13-1161 split into analysts and marketing specialists): they are not hand-checked here either, and are labels and opt-in filters. Only reliable rules hide by default (seniority, likely ghosts, occupations that aren't a tech path); every rule reports how many listings it would hide. **It builds no matching models:** résumé-to-listing fit (keyword overlap, TF-IDF, BGE embeddings, RRF) is in `ML_Fit`, which uses this notebook's cleaned text and skills. A text feature is kept only if it adds information beyond the non-text features.

**For career changers,** the most important text features are what a listing asks of a newcomer (experience, education, licenses, training) and which of its skills carry over from other fields.

**Inputs (the Lean chain):** `EDA_SDA_1_Lean`'s in-scope table (`analysis_table.parquet`) and
`EDA_SDA_2_Lean`'s rule flags (`scope_flags.parquet`), joined by `id`, plus the description text for those
listings (`scope_descriptions.parquet`). All three are in `data/clean/lean/`.

**The series** (each notebook hands its final table to the next):

| Notebook | Question | Hands off |
|:--|:--|:--|
| EDA_SDA_1 | What's in the data, how does tech compare to other fields, and what's in scope? | Scope table, descriptions, occupation summary, company activity |
| EDA_SDA_2 | Which in-scope listings do we keep, tag or hide? | Scope table, non-text features |
| **EDA_SDA_3** | **What do the descriptions say? Skills mapped to O*NET** | **Scope table + text features** |
| EDA_SDA_4 | Which features hold up? Statistics, confounders, the adjusted Cox check | Modeling-ready table |
| TimeSeries | When do listings appear and close? | Timing features |
| Profiles | Who really changed into or within tech or math, and how? Career-switch rules on the public résumé data (frozen) | The 2,813 career changers, and the table ML_Fit trains on |
| ML_Fit | Which roles and listings fit a career changer? Keyword → TF-IDF → BGE → RRF, skills-based matching, a trained layer on real career moves | Fit scores, role discovery, the compass |
| ML_Rank | How should listings be ranked? The listing timing model, combined with fit into the priority queue | The priority queue algorithm |
| Evaluate | How well do the app's rankers do on real career moves? `rrf`, `mlrank` and the combined model | The ranker comparison |

**Standard:** the TripleTen course templates are the baseline (structure, methods, reporting). Every applicable course method is used, and taken further with more advanced techniques where the data calls for them.

**View:** listings with a description (the share 1b reports from `has_desc`). Text features are **unknown** for the rest, never "no", and "has a description" stays a standard confounder: listings without one close much faster.

**Data rules:** aggregates only; never print descriptions, titles or sample rows, except in cells tagged `data-rows` (review cells for a teammate to check locally; cleared before every commit). Text stays local: no listing text goes to outside AI services. Clear outputs before committing.

### Clarifying questions
- What does description length actually capture: employer intent, or boilerplate?
- Does urgency wording, read in context rather than by keyword, relate to how long listings stay open? (EDA_SDA_1_Lean left it here: its keyword flag misfired and showed no gap.)
- Can evergreen wording ("always accepting applications") identify ghost listings?
- Are EDA_SDA_1_Lean's career-change flags (years required, degree required, training and career-changer wording) accurate, and what do they miss? Two are urgent: EDA_SDA_1_Lean's "asks 5+ years" rule would hide a large share of open listings if a user turned it on, and its degree flag fires on very few (both measured here in 3a and 3b, and tagged in EDA_SDA_2_Lean 4.5 as `asks_5_plus_years` and `degree_no_equivalent`).
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

## Contents

- [Purpose](#purpose)
  - [Clarifying questions](#clarifying-questions)
  - [Roadmap](#roadmap)
- [Plan: text features to build](#plan-text-features-to-build)
- [Plan: preprocessing](#plan-preprocessing)
- [2. Description cleanup](#2-description-cleanup)
- [3. The blocker rules: years of experience and degree](#3-the-blocker-rules-years-of-experience-and-degree)
- [4. Other text features](#4-other-text-features)
- [5. Validation samples (not being marked)](#5-validation-samples-not-being-marked)
- [6. Skills mapped to O*NET](#6-skills-mapped-to-onet)
- [7. Final table](#7-final-table)
- [8. Conclusion](#8-conclusion)

## Plan: text features to build

| Feature | Question | Notes |
|:--|:--|:--|
| **Description cleanup** (moved here from EDA_SDA_2 2b, which EDA_SDA_2_Lean leaves out) | Strip HTML and boilerplate (equal-opportunity statements, benefits, "about us") | `desc_clean` and `desc_words_clean`; the base for every feature below and for `ML_Fit` |
| Description length | Is it employer intent, or just boilerplate? | Measured on `desc_clean` |
| Urgency wording | Does real urgency (in context, not "urgent care") go with shorter or longer windows? | EDA_SDA_1_Lean's keyword flag showed no gap; may mark hard-to-fill roles |
| Remote mentions | Does remote add beyond `contract_types`? | |
| **Evergreen wording** | Listings that collect applications instead of filling one job: a ghost-listing signal | Six phrase groups, each scored and checked on its own: **talent pool** ("join our talent community", "future opportunities"), **no specific opening** ("general application", "not for a specific position"), **always hiring** ("always accepting applications", "rolling basis"), **many openings in one ad** ("multiple openings", "various locations"), **pipeline labels** ("pipeline requisition", "evergreen requisition"), **staffing-style** ("on behalf of our client"); plus a generic-title check ("Software Engineer – All Levels"). Compared with EDA_SDA_2_Lean's likely-ghost flag (3.1): kept only if it adds to "old and from a quiet company" |
| Hiring-difficulty phrases | "Sign-on bonus", "hiring immediately", "relocation assistance", "will train" | Pairs with reposts and listing age |
| Years of experience required | "3+ years": is EDA_SDA_1_Lean's rule accurate? Required or preferred? In the field, or any experience? | The most common blocker for career changers |
| Education required | "Bachelor's required" vs. "or equivalent experience"; field-specific degrees | EDA_SDA_1_Lean's pattern only fires when "required" comes within six words of "bachelor's", so the most common wording ("Bachelor's degree in Computer Science or a related field") never matches, and it flags only a small fraction of listings (3b measures both here). Rebuilt in section 3, with requirements and preferred sections told apart |
| **Clearance, sponsorship, travel, shifts** (added 2026-10-10) | Security clearance required; "no visa sponsorship"; travel percentage; shift or on-call work; contract-to-hire | Clearance and sponsorship are blockers for some career changers: a label on the listing and an opt-in filter in the app, like a missing degree (never hidden by default) |
| Licenses and certifications required | "Security+", "CCNA", "CISSP" and the like | A required credential the career changer lacks: a label and an opt-in filter |
| Training and career-changer wording | "Will train", "paid training", "career changers welcome", "transferable skills" | Reports EDA_SDA_1_Lean's flags as keyword rules with their firing rates (the hand-check sample is not being marked, decided 2026-10-13) |
| **Text-rule filters** | Which years-asked threshold and which required credentials would a user filter on, per role bucket? | Not validated (decided 2026-10-13): saved in this notebook's final table as labels, with a count of the listings each would hide if turned on (opt-in in the app, never hidden by default, never removed) |
| Required vs. preferred qualifications | Split on headings like "Requirements" and "Nice to have" | Part of fit and gap analysis in `ML_Fit` |
| **Skills, mapped to O*NET** | Matched as whole phrases against O*NET's tools and skill names on requirement lines (PredictLeads `tags` are sparse), section 6 | `ML_Fit` matches résumés to listings on these skills, so "transferable" doesn't depend on the old field's vocabulary |

**Moved to `ML_Fit`:** role similarity, TF-IDF and embeddings for matching. **Moved to `ML_Rank`:** TF-IDF terms that predict how fast a listing closes (a text input to the timing model). Topic modeling stays a stretch goal.

## Plan: preprocessing

Course standard (MLforTexts): normalize text (lowercase, regex to keep letters), remove NLTK stopwords, lemmatize with spaCy. Advanced: remove boilerplate where headings make it detectable. Any vocabulary learned from the text is fit on the training months only.

**Evaluation (course standard):** the plan was precision and recall on a hand-checked random sample against EDA_SDA_1_Lean's keyword rule. **Decided 2026-10-13: the samples will not be hand-checked,** so each rule is reported as a keyword rule with its firing rate, against EDA_SDA_1_Lean's rule where one exists, and not as a validated blocker.

## 2. Description cleanup

**Why:** every text feature, and `ML_Fit`'s matching, should read what the listing asks for, not the legal and benefits text most listings share. At full scale the descriptions are several GB of text, so **the cleanup runs batch by batch** (`src/text_features.py`): each batch is cleaned, its features computed and both written to disk, then dropped from memory.

| Step | What | Output |
|:--|:--|:--|
| 2a | HTML to text, one line per paragraph, list item or heading (long unbroken text split into sentences) | Lines per listing |
| 2b | Remove boilerplate line by line: known patterns (equal opportunity and legal, benefits, application logistics), "About the company" sections, and text shared word for word across 5+ employers (learned on a sample of April–July listings; lines with a requirement cue are never cut) | `desc_clean` (`eda3_desc_clean.parquet`), `words_clean` |
| 2c | Course standard: lowercase, letters only, NLTK stopwords, spaCy lemmas, on a stratified sample | `desc_lemmas` (for word counts and TF-IDF later) |
| 2d | Check the text really is English: the scope cut only trusts the provider's `language` label, so each description's English function words are counted against five other languages' | `not_english` (`eda3_language.parquet`) |

The rules in sections 3–4 run on the cleaned text, which keeps case, numbers and punctuation ("3+ years", "B.S.", "Security+"), and tell **requirement lines from preferred ones** using the headings kept in 2a.

**What the data shows**
- **The cleanup ran on all 878,937 scope listings, 82.2% of which (722,602) have a description.** In this run the text features were loaded from the cache (`eda3_text_features.parquet`), so no run time or template-line count is printed.
- **Boilerplate removed, by kind (words, share of all description words):** equal opportunity and legal 35.0M (6.6%), about the company 23.3M (4.4%), benefits 23.2M (4.4%), shared template text 19.8M (3.7%), application logistics 2.4M (0.4%).
- **Listings keep most of their text:** the median listing keeps 85-86% of its words in every source type (company site 717 to 557 words, hiring platform 785 to 619, job board 621 to 499), and none is cleaned to nothing.

**What the data shows**
- **Words per description fall at each step:** a median of 736 after HTML removal, 573 after boilerplate removal and 395 as lemmas with stopwords removed (the lemma step is on a 150,000-listing sample). Means are 734, 590 and 406.
- **The longest descriptions are long before and after:** the 90th percentile is 1,150 words after HTML removal and 959 after boilerplate removal, and the maximum (7,827 words) is the same at both steps.
- **The scope is almost all English.** Of the 715,227 descriptions with 30+ words, a median 99.4% of function words are English. 2,487 listings (0.28% of the scope, 0.34% of those with a description) read as not English: 2,208 where another language's function words win and 299 with over 10% of characters outside Latin-1; all carry the provider's `en` label. Another 753 are borderline (worth reading by hand), and 163,710 (18.6% of the scope) have under 30 words and are not scored. The flagged share by role bucket runs from 0.17% (security) to 0.39% (software development and networks and systems).

## 3. The blocker rules: years of experience and degree

EDA_SDA_1_Lean's rules read every line the same way: "3+ years required" and "5 years preferred" both count, and its degree rule only fires when "required" comes within six words of "bachelor's", so the most common wording ("Bachelor's degree in Computer Science or a related field") never matched, and it fires on very few listings (3b measures both here). Here both are rebuilt on the cleaned text, with **requirement lines told apart from preferred ones** (from the headings kept in 2a, and lines that say "preferred" or "a plus"):

- **Years:** `years_asked` (any line, like EDA_SDA_1_Lean), `years_required` (requirement lines only), `years_preferred`. Also reads "minimum of two years" and number words.
- **Degree:** `degree_required` (a degree named on a requirement line), `degree_preferred`, `degree_equivalent` ("or equivalent experience", "in lieu of a degree"), `degree_level` (the highest level required).

The question for the app: **how many listings would each rule hide if a user turned it on** (too many years required; a degree they don't have). These are unvalidated keyword rules (decided 2026-10-13), so they label listings and drive opt-in filters; nothing hides by default.

**What the data shows**
- **The chart:** the requirement-line rule finds 5+ years in more listings than EDA_SDA_1_Lean's rule in every role bucket, by 0.6 points (database) to 8.6 points (other computer roles, 4.1% to 12.7%); the order barely changes, from IT project management (49.1% against 46.8%) at the top to other computer roles (12.7%) and IT support (13.6% against 11.3%) at the bottom.
- **EDA_SDA_1_Lean's years rule was too narrow:** it finds 5+ years in 34.2% of listings with a description, the rebuilt rule in 39.6% (any line). The new rule finds more years in 11.8% of listings and fewer in 1.2%; both say no years in 33.6% and both find the same number in 52.5%.
- **Required vs. preferred matters a little for years:** 5+ years on a requirement line is 36.2% of listings (36.9% of those open on September 30), and 3.5% ask it only on preferred lines. A filter on 5+ years would hide about 37% of open listings if a user turned it on, so the threshold has to be chosen per role bucket, not once.
- **The years asked climb with the bucket:** the median required is 2 years in other computer roles, 3 in IT support, 4 in business and data analysis, 5 in seven buckets and 6 in IT project management; 5+ required runs from 12.7% (other computer roles) and 13.6% (IT support) to 49.1% (IT project management), and 2 years or fewer from 24.3% (other computer roles) and 22.6% (IT support) down to 5.1% (IT project management).
- **EDA_SDA_1_Lean's degree rule fired on 3.8% of listings; the rebuilt rule finds a degree on a requirement line in 52.6%,** another 10.7% only on preferred lines, and 63.3% anywhere. 29.2% accept "or equivalent experience", and 29.0% require a degree with no equivalent (the "I have no degree" filter).
- **Most requirements are a bachelor's:** of the highest level named on a requirement line, 34.2% are bachelor, 9.6% master, 3.7% PhD and 0.9% associate; 51.5% name none.
- **The degree blocker is lowest in other computer roles (13% with no equivalent) and IT support (20%), and highest in quality assurance and data science (37% each),** with business and data analysis at 32%.
- **These are unvalidated keyword rules:** the section 5 samples will not be marked (decided 2026-10-13), so these rates are firing rates, not precision. They label listings and drive opt-in filters; nothing hides by default.

## 4. Other text features

Each feature is a keyword rule on the cleaned text (`src/text_features.py`), reported with its firing rate and not hand-validated (decided 2026-10-13); it labels listings and can be an opt-in filter, and hides nothing by default. For each: how often it fires, and whether it relates to how fast listings close **beyond the confounders** (the 7-day close rate's gap with vs. without the feature, within company volume band × source type cells, on original listings with a full week of follow-up; EDA_SDA_4 tests the survivors properly). Listings where a feature is unknown (NA) are left out of that feature's comparison, not counted as "without"; the `known` column in 4b shows each feature's coverage (1 = every listing in the comparison is known).

| Group | Features |
|:--|:--|
| Labels and opt-in filters for some applicants | `clearance_required` (vs. `clearance_obtainable`), `no_sponsorship`, `us_citizen_required`, `cert_required` and named certifications, `shift_or_oncall`, `travel_pct`, `contract_to_hire` |
| What a newcomer is offered | `entry_wording`, `changer_wording` (now also bootcamp graduates, self-taught) |
| Hiring difficulty and urgency | `sign_on_bonus`, `hiring_immediately`, `relocation_offered`, `urgency_context` ("urgent care" no longer counts) |
| Evergreen wording (ghost signal) | six phrase groups and `evergreen_any`, compared with EDA_SDA_2_Lean's likely-ghost flag |
| Remote and length | `remote_mention` (minus "remote patient monitoring" and the like), `words_clean` |

**What the data shows**
- **Most features are rare.** Remote is mentioned in 29%, a certification in 12%, a required clearance in 10.0% and no sponsorship in 6.9%; shift or on-call work 5.3%, US citizenship required 4.9%, entry-level wording 3.8%. Career-changer wording is on only 0.46% of listings, and a required certification on 0.93%.
- **Role buckets differ where you would expect:** certifications are mentioned in 40% of security listings (CompTIA 17%, security certifications 33%), 22% of IT support (CompTIA 17%) and 20% of IT project management (project certifications 15%); clearance is required in 26% of security, 15% of quality assurance and 14% of networks and systems listings, but 1.3% of technical sales. Shift or on-call work is 13% of IT support and 15% of other computer roles. Remote is mentioned least in other computer roles (11%) and most in technical sales (39%).
- **The 7-day close-rate gaps are modest.** About 21% of the original listings close within a week. Within company volume band x source type cells, the gaps run from -5.6 points (always-hiring wording, 2.2% of listings) to +8.6 points (urgency in context, 0.2%). Next on the negative side are sign-on bonus (-3.7), any evergreen wording (-3.0), Cisco certifications (-2.9) and a required degree (-2.9); on the positive side, no-specific-opening wording (+3.6), staffing-style wording (+3.0) and entry-level wording (+2.5). Features on 0.5% of listings or fewer are noisy.
- **Some raw gaps change sign once the confounders are held fixed:** contract-to-hire goes from +4.2 to -1.0 points and no-opening wording from -4.2 to +3.6. This is a first look, not a causal estimate; EDA_SDA_4's adjusted model decides which survive. In this run no listing with a description has an unknown feature, so the `known` share is 1 for every row.
- **Evergreen wording does not line up with the ghost flag:** any evergreen phrase is on 4.1% of closed listings, 3.3% of likely ghosts, 4.5% of listings open 90+ days from companies still posting and 4.3% of open listings 90 days or younger. As it stands it does not add to "old and from a quiet company"; always-hiring wording (2.1% of listings) is the largest of the six groups.

## 5. Validation samples (not being marked)

**Decided 2026-10-13: these samples will not be hand-checked.** 5a still writes a **sample** for each rule: 60 listings where it fires and 60 where it doesn't, stratified by role bucket, to `data/clean/eda3_validation/<rule>.csv` (local only), and 5b would read marks back and report precision and recall if anyone ever fills them in. Nobody is marking them, so **the rules are reported as keyword rules with their firing rates (sections 3, 4 and 7), not as validated blockers.** That sets what each rule may do: the text blockers (years asked, degree required with no equivalent, clearance, no sponsorship, required certification and the other career-change flags) are **labels on each listing and opt-in filters a career changer turns on, never hidden by default**. Only reliable rules hide by default (seniority, likely ghosts, occupations that aren't a tech path, all decided in EDA_SDA_2). Every rule reports how many listings it would hide (7a). Listing text never leaves the laptop.

**What the data shows**
- **No rule is validated, and none will be:** the samples are written (about 120 rows per rule, for ten rules: 5+ years required, degree required, degree equivalent, clearance, no sponsorship, required certification, entry wording, career-changer wording, evergreen wording, urgency) but are not being marked (decided 2026-10-13), so 5b reports no precision or recall.
- **So the rules are keyword rules reported with their firing rates:** they are labels and opt-in filters, never hidden by default; the 90% precision bar that would have let a rule hide listings is not applied to any of them.

## 6. Skills mapped to O*NET

**Question:** which tools and skills does each listing ask for, and how do they line up with what O*NET says the occupation needs? The answer feeds `ML_Fit`'s skills-based matching and the retraining compass's "skills that carry over".

**What O*NET is.** The **Occupational Information Network** is the US Department of Labor's public description of work: for every occupation in the O*NET-SOC classification it records the skills, knowledge and abilities the occupation needs, the software and technologies people use in it, how much preparation it takes (its **Job Zone**, 1 little or none to 5 extensive), and which occupations are related to it. The ratings come from surveys of workers in the occupation and from occupational analysts, rated for importance (`IM`, 1 to 5) and for level (`LV`, 0 to 7), and they describe **the occupation in general, not these listings**; there is no pay in it (pay always comes from the listings). It is keyed by the same O*NET-SOC code PredictLeads puts on every listing (`EDA_SDA_1_Lean`, Purpose), which is what lets the two be joined at all.

**Where it comes from.** The **O*NET 31.0 Database** (August 2026 release) from the O*NET Resource Center: tab-separated tables, public, no account or key. `python -m fetch.get_onet` downloads it into `data/external/onet/` (git-ignored) and keeps the nine tables this project reads (and its Read Me), about 10 MB - Occupation Data, Essential Skills, Transferable Skills, Software Skills, Job Zones and Job Zone Reference, Related Occupations, Content Model Reference and Scales Reference. The license is **CC BY 4.0**, which asks for credit and for changes to be stated: the credit "O*NET 31.0 Database, USDOL/ETA" travels with every result, and the files are kept as downloaded - `src/onet.py` only reads them, keeping the importance (`IM`) and level (`LV`) rows and dropping the ones O*NET marks suppressed or not relevant. Release 31.0 has no file called "Skills" or "Career Changers Matrix": those are Essential and Transferable Skills, and Related Occupations. 6a prints how many occupations carry skill ratings.

**What it is for here.** The listings say what each employer asks for; O*NET says what the occupation needs. Side by side they give four things the listings alone cannot: a **vocabulary** of real tool and skill names to find in the text (6a-6c), a **check** on whether a listing asks for what its own occupation needs (6d), an **accessibility prior** from Job Zone (6e), and the **skills that carry over** from a prior career to a target role, which is the compass's transfer side (6f, then `ML_Fit` 5a and 8b).

| Step | What | Output |
|:--|:--|:--|
| 6a | **Skill vocabulary:** every software tool O*NET lists (the Workplace Example, with its hot-technology and in-demand flags) plus the skill element names, lowercased with punctuation removed and common aliases folded together ("js" to "javascript", "MS Excel" to "Microsoft Excel"; C++, C# and .NET keep their meaning). Matched as **whole phrases**, so "java" is not found inside "javascript". Everyday words that are also tool names ("Act!", "Analyze", "Google" alone, "Tanks", "Reduce", "Sand", "Freight+", "Route", "Dig", "Word", "Access", ...) and the generic one-word skill names (Science, Writing, Speaking, Mathematics, Monitoring, Coordination, Persuasion, Instructing, Installation, Repairing) are left out, because matching them as phrases fires on ordinary sentences; "R", "Go", "C" and "Swift" are matched only in context | The vocabulary and a compiled matcher |
| 6b | **Extraction from requirement lines only** (a line under a requirements or preferred heading, or with a requirement cue such as "years", "experience", "proficiency"; responsibilities are not read), batch by batch from `eda3_desc_clean.parquet` with the same crash-safe `.part` writes as 2a-2b. Each match is marked **required** (on any required line) or **preferred** | `eda3_skills.parquet`: `id`, `name`, `kind` (tool or skill), `section`, `element_id` (cached; delete to rebuild) |
| 6c | **Coverage** and the most-asked tools and skills per role bucket | Aggregates |
| 6d | **Agreement with the listing's own occupation:** the share of a listing's matched skills that O*NET lists for its O*NET-SOC code (tools on O*NET's Software Skills list for the code; skills rated 3+ of 5 in importance, Essential or Transferable) | Aggregates |
| 6e | **Job Zone per bucket** (O*NET's 1-5 scale of preparation needed) as an accessibility prior | Aggregates |
| 6f | **Transferable skills that carry over** from each of Profiles' prior-career areas (`BACKGROUND_AREAS`, as O*NET groups in `BACKGROUND_SOC`) to each target bucket: the compass's "skills that carry over"; in the app, a user's own résumé is matched the same way (`ML_Fit`) | Aggregates |

Limits: a skill element name such as "Programming" or "Time Management" can still be ordinary wording, so skill matches are looser than tool matches; the keyword rules are not hand-checked (as in section 4), so they are labels and matching features, never filters that hide listings.

**What the data shows** (O\*NET 31.0 Database, USDOL/ETA, CC BY 4.0; keyword matching on requirement lines)
- **Coverage:** 446,327 listings with a description (of 878,937 in scope) have at least one matched tool or skill, 1,484,771 matches in all, 82% of them on required lines. Across all buckets, 62% of listings have a match, 56% on a required line and 53% a tool; the median listing with any has 2 matches (90th percentile 7). Highest are software development (71%), quality assurance (70%) and data science (67%); lowest are technical sales (43%), other computer roles (44%) and IT project management (46%). Tool coverage runs from 67% (software development) and 64% (quality assurance) down to 27% (other computer roles) and 28% (technical sales). The vocabulary is 8,741 tools (174 hot technologies, 284 in demand) and 25 skill names, with 22 skill names left out (very short, everyday or generic words) and 4 (R, Go, C, Swift) matched only in context; 910 O\*NET occupations carry skill ratings. Of the tool mentions, 79% are hot technologies and 79% in-demand ones.
- **Most-asked tools** differ by bucket: Python is in 37% of data science listings, 32% of quality assurance, 30% of software development, 24% of security and 21% of database, against 5.6% of business and data analysis and 3.9% of IT support. Tableau is in 16% of data science and 7.4% of business and data analysis listings; Linux in 13% of quality assurance and security, 10% of networks and systems and software development, 6.0% of IT support; Kubernetes in 14% of software development; Snowflake in 9.9% of database and 7.5% of data science. With the everyday-word names excluded, the top entries are recognisable tools (Python, Linux, Kubernetes, JavaScript, C++, Git, Docker); the category entry "Programming languages" (4.4% of data science, 6.0% of quality assurance) is a catch-all name, and C and Go (matched only in context) are undercounted (see the limits).
- **Most-asked skills:** Troubleshooting is in 42.5% of IT support listings (23% of other computer roles, 20% of networks and systems); Programming in 23% of quality assurance and 22% of software development; Time Management in 9.2% of business and data analysis and 7.4% of IT support; Critical Thinking in 5.0% of data science and 4.8% of business and data analysis; Negotiation in 5.3% of technical sales listings. Outside these, skill names are rare (most are under 3% in every bucket).
- **Agreement with the listing's own occupation:** the measure is the share of a listing's matched tools that are on O\*NET's Software Skills list for its code, and of its matched skills that O\*NET rates 3+ of 5 in importance for it (Essential or Transferable Skills). The median listing has 100% of its matched tools and skills on the list for its own code, and the median is also 100% for tools alone and skills alone. A median of 100% partly reflects that the median listing has only 2 matches, so a listing is mostly all on the list or not. Buckets differ: technical sales is 50% (tools and skills, tools only and skills only alike), other computer roles 0% on all three, IT project management 67%, database 72% (tools only 80%, skills only 50%), security 75% (tools only 88%, skills only 50%), networks and systems and quality assurance 83% (tools only 100% for quality assurance); business and data analysis, data science, IT support and software development are 100% on all three. 13.8% of the 446,327 listings with a code and a match have none of their matches on the list for their code.
- **Job Zone** (O\*NET's 1-5 preparation scale, by the listing's occupation code): IT support is all zone 3 (mean 3.0); networks and systems is 58% zone 3, 37% zone 4 and 5.4% zone 5 (3.5); software development 24% zone 3 and 76% zone 4 (3.8); data science 69% zone 4 and 31% zone 5 (4.3); the other six buckets are all zone 4. 0.8% of listings have no Job Zone, and other computer roles has no row in the table. The zone describes the occupation, so buckets that sit on one or two codes show a single value.
- **Skills that carry over** (transferable skills rated 3+ of 5 in both a prior-career area and the target bucket; areas are Profiles' public-résumé backgrounds, grouped as O\*NET codes): Complex Problem Solving is the top transferable skill of 5 of the 10 buckets that have ratings (3.51 to 3.88), Programming of software development (4.03) and quality assurance (3.62), Coordination of IT project management (4.00), Systems Evaluation of networks and systems (3.80) and Persuasion of technical sales (4.12). The number of shared skills is largest from self-employed and small business and from business, marketing and management (12 into IT project management, 11 into technical sales, 9 into business and data analysis), and from humanities and social sciences (11 into IT project management and technical sales); it is 5 or fewer into data science, database, security and software development from every area. Food and hospitality shares 0 to 2 skills with any bucket, logistics and trades 1 to 2, and agriculture and manufacturing 0 or 1 (other computer roles aside). Other computer roles shows 0 shared skills from every area (it has no transferable-skill ratings in this run). Most areas share Complex Problem Solving and Judgment and Decision Making; customer service shares only Time Management with data science, quality assurance and security, Coordination with database, and Time Management and Coordination with networks and systems.
- **Limits:** this is keyword matching on requirement lines, not understanding. Everyday-word tool names (Act!, Analyze, Google alone, Tanks, Reduce, Sand, Freight+, Route, Dig) and generic one-word skill names (Science, Writing, Speaking, Mathematics, Monitoring, Coordination, Persuasion, Instructing, Installation, Repairing) are excluded; skill names that remain (Programming, Time Management) can still be ordinary wording, so skill matches are looser than tool matches. R, Go, C and Swift are matched only in context, so they are undercounted. O\*NET describes occupations, not employers, so agreement and Job Zone say what the occupation code lists, not what a given employer needs. The rules are not hand-checked (section 5), so they are labels and matching features, never filters. Backgrounds come only from Profiles' public résumés; the team's own résumés and backgrounds are not used. Associations only, no causal claims.

## 7. Final table

One row per scope listing (`id`), the text features from sections 3–4, and **text-rule flags** for the app (the `blocker_*` columns). They are unvalidated keyword rules (decided 2026-10-13): the app shows them as labels and opt-in filters (for example "I have no degree"), never hides by default, and 7a reports how many listings each would hide. Blank means unknown: the listing has no description. Saved as `text_table.parquet` for EDA_SDA_4 (joined by `id`) and `ML_Fit`.

## 8. Conclusion

**Summary**
- **Cleanup:** 878,937 scope listings, 82.2% with a description; the median description falls from 736 words to 573 after boilerplate removal (85-86% of text kept per source type). Equal opportunity and legal text is the largest share removed (6.6% of all words).
- **Rebuilt rules (unvalidated keyword rules):** 5+ years on a requirement line is 36.2% of listings (36.9% of open ones), against 34.2% for EDA_SDA_1_Lean's rule; a degree is required on 52.6% (EDA_SDA_1_Lean: 3.8%), 29.0% with no equivalent accepted.
- **Other features:** most are rare (remote 29%, certification 12%, clearance 10.0%, no sponsorship 6.9%, career-changer wording 0.46%). Their 7-day close gaps within confounders run from -5.6 to +8.6 points; the largest gaps sit on features that fire on 0.5% of listings or fewer.
- **Evergreen wording** does not concentrate in likely-ghost listings (3.3%, against 4.1-4.5% elsewhere).
- **Skills (O\*NET 31.0, USDOL/ETA, CC BY 4.0):** 62% of listings with a description have at least one matched tool or skill (53% a tool); the median listing with any has 2. Python is in 37% of data science and 30% of software development listings; Troubleshooting is in 42.5% of IT support. The median listing has 100% of its matches on O\*NET's list for its own code (partly because the median listing has only 2 matches), and 13.8% have none on it. Job Zone is 3.0 for IT support, 3.5 for networks and systems, 4.3 for data science and 4.0 for most other buckets. Complex Problem Solving is the top transferable skill of 5 of the 10 buckets with ratings; the prior-career areas share the most skills with IT project management and technical sales (up to 12 and 11) and 5 or fewer with data science, database, security and software development.

**Answers to the clarifying questions**
- *Length: intent or boilerplate?* Not answered here; the cleanup shows boilerplate is a large part of the text, and length (`words_clean`) is saved for `EDA_SDA_4`.
- *Urgency in context vs. close time?* It fires on 0.20% of listings, with a +8.6-point gap within confounders: too few listings to rely on.
- *Evergreen wording as a ghost signal?* Not as it stands (see above).
- *Are EDA_SDA_1_Lean's flags accurate?* The years rule misses some (the new rule finds more years in 11.8% of listings) and the degree rule missed most requirements (3.8% against 52.6%). Precision is untested: the hand-check samples will not be marked (decided 2026-10-13).
- *Which requirements hide a listing?* None by default. Years asked, clearance, no sponsorship, a required certification and a degree with no equivalent are **labels and opt-in filters**, and would hide this much if a user turned each on: 5+ years required (36.2% of listings), clearance (10.0%), no sponsorship (6.9%), a required certification (0.9%) and a degree with no equivalent (29.0%). **None is validated, so none hides by default.**
- *Skills mapped to O\*NET?* Yes, by keyword matching on requirement lines: 1,484,771 matches over 446,327 listings (section 6), saved as `eda3_skills.parquet`. Everyday-word tool names and generic one-word skill names are excluded; the matches are still unchecked keyword hits, which a check should look at before they feed matching.

**Limits:** the gaps in 4b are a first look and associational, not causal; the rules are unvalidated keyword rules (firing rates, no precision); the O\*NET skills are keyword matches (looser for skill names than for tools; R, Go, C and Swift only in context; occupation-level, not employer-level); other computer roles has no Job Zone or transferable-skill ratings in this run; and the length question is open.

**Handoff** (all in `data/clean/lean/`, the Lean chain's folder): `text_table.parquet` (878,937 listings, 45 columns: `id` + the text features and the text-rule labels, blank where there is no description) for `EDA_SDA_4` and `ML_Fit`, plus `eda3_desc_clean.parquet` (cleaned text), `eda3_lemmas_sample.parquet` (150,000 listings) and `eda3_skills.parquet` (`id`, `name`, `kind`, `section`, `element_id`; 1,484,771 matches over 446,327 listings) for `ML_Fit`.
