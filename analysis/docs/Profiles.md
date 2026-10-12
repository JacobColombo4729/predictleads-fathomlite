<!-- Generated from notebooks/Profiles.ipynb by src/export_markdown.py (text only; no code). Regenerate after editing the notebook; don't edit by hand. -->

# Job Listing Prioritization for Career Changers: Profiles (career switches)

PredictLeads × TripleTen externship · Jacob Colombo

## Purpose

Finds **real career changers** (anyone switching from their current background to a different role or field) in a public resume dataset: people who moved from another field into a **computer and mathematical** role, or from one tech role to another. It feeds two things:
- **The retraining compass, past side** (`DIFFERENTIATORS.md`): prior career → the training people took → their first tech job (4a.2), and tech job before → after a move within tech, sideways or up (4a.3). EDA_SDA_2 adds each role's 2026 openings, pay, closing speed and expected growth.
- **The role model** (Option B, the 2026-10-09 entry in `DECISION_LOG.md`): each career changer gives a **pre-switch profile** (jobs and education before the switch) and a **real label** (the role they moved into).

**How a career changer is found.** Each resume is read as one timeline: jobs and education in date order, each entry labelled tech/math, other field, unclear, or skip (internships, high school). The **original background** is everything before the first tech or math entry. **Nobody who reached a computer and mathematical job is thrown away:** each person gets labels (origin, retrained, first tech job level, every tech role reached), and `status` names the main one. **Career changers are a view,** in three pools: **retrained after working in another field** and **retrained as a new grad from another field** (another field, then a tech or math program, then the tech job), and **changed roles within tech** (a move into a different tech role bucket after 12+ months in the first). Any job level counts, including manager. The ML model can learn from everyone, with origin as a feature; results are reported on career changers. The rules live in `src/career_switch.py`.

**Data:** Kaggle `suriyaganesh/resume-dataset-structured` (MIT license, version 2 of 2024-11-14): about 54,000 resume records, converted by OCR and an LLM, so expect extraction errors. Stored in git-ignored `data/external/resume_structured/`; download it with `python -m src.get_resumes` from `analysis/` (no Kaggle account needed).

**Data rules:** names and contact details (`people.csv`), employers and locations are never loaded. Aggregates only, except the cells tagged `data-rows` (the 4a.2d and 5b previews, the 4a.2e title list and the 4c review sample): clear them before committing. Never paste rows into AI tools.

**Leakage rule:** the pre-switch profile uses only jobs and education that started before the switch. Undated programs are placed by degree level (flagged in `study_undated`). Skills aren't dated, so they stay out of the pre-switch profile.

### Clarifying questions
- How many people switched into a computer and mathematical role, by retraining or by moving within tech?
- From each prior career, which first tech jobs did people reach, and through which training? (The compass for changing careers.)
- Within tech, which moves do people make, and how many are a step up rather than sideways? (The compass for moving across or up.)
- How often do the rules mistake a title change, a student job or a technical background for a career change?

### Roadmap
1. Setup · 2. Load, then remove duplicate resumes (2b) · 3. Dates and title flags
4. Find the switches; viable candidates (4a.1); retrained: prior career, training and first tech job (4a.2); changed roles within tech, sideways or up (4a.3); the review round (4b, 4c)
5. Freeze and save: the modeling table · 6. Conclusion

## 2. Load

Only the columns the rules need. If the dataset's headers differ from its documentation, this stops and lists what's missing.

### 2b Duplicates

The dataset combines several resume collections, and the first run's counts were almost all multiples of 3, which suggests each resume appears about three times under different `person_id`s. Duplicates would show the same person twice in a review round and put copies of one person on both sides of a train/test split (leakage).

Each person gets a **fingerprint**: every job (title, start and end date as written) plus every program (name and start date), sorted. People with the same fingerprint are the same resume; the lowest `person_id` is kept. Only **exact** copies are caught: a copy with an OCR difference gets a different fingerprint.

**What the data shows**
- **54,933 resume records held 17,952 unique people:** 36,981 were duplicates, so each resume appeared about 3 times.

## 3. Dates and title flags

Dates become months (`MM/YYYY`; a bare year becomes January; "Present" becomes the dataset's update month, 2024-11). Titles are matched against the rules: which computer and mathematical bucket they fall in, and whether they're senior, school or internship work, or already technical. 3b shows the *patterns* of dates the parser still can't read (# for a digit, Word for letters), so the parser can be extended without looking at any dates.

## 4. Find the switches

For each person, in date order: their **first computer and mathematical job** is the switch, and the non-computer-and-mathematical jobs before it are their **background**. Everything before it is read as one timeline with their education. Their **original background** is everything before their first tech or math entry (a tech job, or a tech or math program). A person is a **candidate** if that background includes another field **and** they trained into tech or math before the switch: `candidate: retrained, new grad from another field` or `candidate: retrained, after working in another field`. Each rejection is kept with its reason, so we can review the rejected pools too.

The rules label instead of rejecting: every person who reached a computer and mathematical job stays in `switches`, with `origin`, `retrained`, `first_level` and `roles_reached` (every tech role they held, not just the first). 4a's second cell counts people by label.

**What the data shows**
- **Almost everyone has a readable job date** (17,950 of 17,952), up from 63% before the date fix.
- **237 career changers moved into data science: 218 by changing roles within tech, only 19 by retraining.** That's why the retraining compass (4a.2b) shows almost no data science.
- **16,237 people (90%) held a computer and mathematical job;** 2,813 of them are career changers (4a.1). Counting IT project management as tech (since 2026-10-10) added about 1,100 people who reached tech.

### 4a.1 Viable candidates

How many people the rules leave, and who they are in numbers. **Viable** means they passed the rules; the rules were spot-checked on random samples (4b/4c), not every person. Counts are of unique people (after 2b). Aggregates only.

**What it shows**
- **2,813 career changers (16% of unique people):** 2,049 changed roles within tech, 407 retrained after working in another field and 357 retrained as new grads from another field.
- **The biggest drop is from the 16,237 who reached tech:** most didn't change field or role bucket in a way the pools count.

**What it shows**
- **The typical switch is in 2014, after 2.4 years in the previous field or role** (medians; half switched between 2010 and 2016). Both are far from a bell curve (skew −1.5 and 3.6), so medians describe them better than means.
- **No strong correlations.** The strongest is −0.31: later switchers list fewer jobs, because their resumes have had less time after the switch. Everything else is within ±0.22, so the features don't repeat each other.
- **People who retrained after working took about 15 months (median) from their last other-field job to their first tech job;** a quarter started within a month, and some started tech work before leaving the old job (negative gaps). A few long tails (up to 55 years in the previous field) stay in; medians aren't moved by them.

### 4a.2 Retrained into tech or math: where they came from, what they trained in, and where it led

One row per viable candidate, stating their **prior career**, the **field they trained in** (grouped by `TRAINING_FIELDS` in `src/career_switch.py`) and the program as written. The full table is saved to `data/profiles/` for browsing; everything printed here is aggregate except the preview in 4a.2d.

**What it shows**
- **Prior career shapes where people land.** Arts and media (52 people), humanities (47) and engineering (47) went mostly into software (about half of each); business (129) spread across networks (24%), software (20%) and **IT project management (17%)**; administration and office (91) across networks, software and IT support; logistics and trades (36) about 40% into networks, and military and public service (29) about 40% into IT support. Overall, first tech jobs were software (30%), networks and systems (22%), IT support (15%), databases (12%), IT project management (10%), security (8%), data science (2.5%) and quality assurance (under 1%).
- **The biggest single paths (% of all 764 retrained people):** computer science → software (75 people, 9.8%), software and web development → software (70, 9.2%), information technology → networks and systems (65, 8.5%) and → IT project management (56, 7.3%), networking → networks and systems (32, 4.2%). Information technology, the biggest field (284), spreads across every role.
- **Only 19 of the 764 retrained people (2.5%) started in data science,** so its share is small in every training field. People reach data mostly by moving within tech (218, 4a.3), so the compass should point career changers to a first tech role on the way to data, not only to a data course.
- **121 people (16%) fall in "other":** their job titles matched no area (4a.2e lists the common ones). They stay in: they're real career changers. The top programs as written are all computer science or information technology and are grouped correctly.

**What it shows**
- **Most retrained switches are from 2010–2019 (571 of 764, 75%);** only one is dated 2020 or later. Switch dates are descriptive only and never filter anyone.
- **Some fields are recent:** 76% of software and web development switches and 59% of cybersecurity switches are from 2015–2019, while information technology and computer science go back further.
- **Retraining shifted toward people who'd already worked:** the odds of being a new grad fall by about 43% every 5 years (× 0.57). From about 2015, most people who retrained had worked in another field first.

**Widening the prior-career areas.** 4a.2e lists the most common job titles (or first fields of study, for new grads) among retrained people whose prior career is "other": no keyword in `BACKGROUND_AREAS` (`src/career_switch.py`) matched them. Tell Claude which career each common title belongs to, in your own words (for example "titles like X are healthcare"), or add keywords to `BACKGROUND_AREAS` yourself; then rerun from 4a. Changing areas never changes who is a career changer.

### 4a.3 Changed roles within tech: across and up

People already in tech who moved into a **different role bucket** after at least 12 months in their first one (for example IT support → data science). For a user who already works in tech, these are the comparable paths: the compass for moving across or up. Each move is also labelled sideways or up, by comparing the level of the new job with the highest level held in the first role. Aggregates only.

**What it shows**
- **2,049 people changed roles within tech.** The largest moves (% of all 2,049): IT support → networks and systems (13.6%), networks and systems → IT support (9.7%; traffic runs both ways), IT support → security (8.0%), developers → databases (6.9%) and → data science (6.7%). **190 moved into IT project management**, mostly from networks and systems (about 70), IT support (about 45) and software development (about 45).
- **218 moved into data science,** mostly from software development (about 140) and databases (about 55). With the 19 who retrained into it, 237 career changers reached data; quality assurance is the smallest destination (39).
- **Most moves are sideways, not up:** 79% stayed at the same level, 8.5% moved up to a senior or manager title, and 12.5% moved to a lower one. The share moving up is similar for every role (8–10%; quality assurance's 17% rests on 36 people). Moves into IT project management look 26% "down" because project managers no longer count as managers, so a manager who became a project manager reads as a step down. Levels come from title words, so read these as rough.

### 4b Review round

Set `POOL` (one of the three career-changer pools, or a rejection status from 4a to check whether a rule throws real switches away), then run 4c. Each run shows **5 random people** from that pool, new ones every time, with each person's **whole timeline** (titles and programs with dates; no employers or locations), every entry's label and the switch marked. The printed seed brings a sample back (`SEED = <number>`). Tell Claude which `person_id`s look right or wrong, and why, in your own words; don't paste the timelines. Aim for **about 20 people per pool** (4 runs).

**What "right" means:**

| Pool | Right if | Common mistakes to look for |
|:--|:--|:--|
| Retrained after working in another field | A real non-tech job (12+ months), then a tech or math program, then a tech job | The "other field" job was really technical; the program came after the tech job; a student job counted as a career |
| Retrained, new grad from another field | Studied a non-tech field first, then a tech or math program, then a tech job | The first degree was really technical; the order of the programs is wrong (undated programs) |
| Changed roles within tech | 12+ months in one tech role, then a genuinely different tech role | The same job retitled; the two roles are really the same bucket; the up/sideways label is wrong |
| A rejection status | The rule was right to reject them | A real career changer the rule threw away |

**Also check:** the prior career area (is "other" hiding a clear career?) and, for within-tech moves, whether "up", "sideways" or "down" matches the titles.

**Spot-check results**

Jacob spot-checked random samples with the checklist in 4b (2026-10-09). The samples looked right, so the rules stand unchanged. Per-pool counts weren't recorded; a future check can add them here (counts only, never the timelines).

## 5. Freeze and save: the modeling table

One table for the `ML_Fit` notebook: **everyone who reached a computer and mathematical job** (labels, not rejections), with **pre-switch features only** and the **labels** the role model predicts. It holds more people than the 2,813 career changers on purpose: the others are extra training examples (for example, people who went straight from a business degree into data show which backgrounds lead there). Results are reported on career changers only (`career_changer`), and the ML notebook tests whether training on everyone beats training on career changers alone. Every column has a role:
- **feature:** known before the switch, so the model may use it
- **label:** what the model predicts, or which people results are reported on (`career_changer`)
- **outcome:** describes what happened at or after the switch; **never a feature**
- **descriptive:** for tables and charts only

Rerun this section after any change to the rules in `src/career_switch.py`.

## 6. Conclusion

**Frozen again on 2026-10-10,** after one rules change: a new **IT project management** bucket (IT project managers are 13% of the scope's listings), and project, program and delivery managers no longer count as "manager or above". The final tables are `data/profiles/modeling_table.pkl` and `data/profiles/retrained_candidates.csv`; rerun the whole notebook after any change to `src/career_switch.py`.

**What we did.** Read 54,933 public resume records as timelines (17,952 unique people after removing duplicates; almost all with readable job dates), labelled everyone who reached a computer and mathematical job (16,237), and found **2,813 career changers** in three pools: 2,049 changed roles within tech, 407 retrained after working in another field, and 357 retrained as new grads from another field.

**What we found.**
- **Where retrained people land depends on where they came from:** software (30% of first tech jobs) and networks and systems (22%) lead overall, but arts, humanities and engineering backgrounds went mostly into software, while administration, business and logistics spread across IT support and networks, and business also into IT project management (17%). Training follows the same split: computer science and software development led to software; networking and cybersecurity to infrastructure; information technology spread across every role, including IT project management.
- **Data science is reached through tech, not straight from a course:** 237 career changers moved into it, 218 of them from another tech role (mostly developers and DBAs) and only 19 by retraining (2.5% of retrained people). The compass should show a first tech role as a step on the way to data.
- **Most moves within tech are sideways:** 79% kept the same level and 8.5% moved up to a senior or manager title, so "moving up" needs its own evidence before it's part of the pitch.
- **Switches take time and have shifted:** the typical switch came after 2.4 years in the previous field (median), people who retrained after working took about 15 months from their last old-field job to their first tech job, and retraining has shifted from new grads toward people who'd already worked.

**Handoff.** The modeling table (section 5) goes to the `ML_Fit` notebook: pre-switch features, the role at the switch as the label, and outcome columns marked. It holds all 16,237 people who reached tech: the 2,813 career changers, on whom results are reported, plus 13,424 others as extra training examples, with origin as a feature. The ML notebook keeps whichever training set predicts career changers' roles better. The prior career and training tables (4a.2b) and the within-tech table (4a.3) are the past side of the retraining compass; EDA_SDA_2 adds each role's 2026 openings, pay and closing speed.

**Limits.**
- The rules were spot-checked on random samples (4b/4c), not every person, and the resumes were converted by OCR and an LLM.
- 16% of retrained people have a prior career of "other" (titles that matched no area), and job levels are read from title words, so the level split is rough.
- The resumes show only people who made a switch, so the tables show where switchers went, never how likely a switch is to work.
- The switches span two decades, so the 2026 listings supply today's openings, pay and timing.
