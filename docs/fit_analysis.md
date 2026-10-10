# Transferable Fit: Approach and Applicant Profiles

How we measure how well a **career changer's** background fits a job listing and which computer and mathematical roles it transfers to (role discovery). A career changer is anyone switching from their current background to a different role or field. This doc also covers how the team builds the applicant profiles it needs. Decision logged in [`decisions.md`](decisions.md). Follow the [Data & Security Rules](../README.md#data--security-rules) throughout.

> **Blocked until the advisor approves collecting applicant profiles.** Don't collect any real profiles before that answer is recorded in the decision log.

---

## 1. Two ways to match a profile to listings

| | **Option A: similarity matching** | **Option B: trained fit model** (chosen) |
|:--|:--|:--|
| **How it works** | Turn the profile and every listing into vectors and rank by similarity: keyword overlap → TF-IDF cosine → BGE embeddings (local) → RRF hybrid of TF-IDF and BGE | A model learns how to weigh many signals into a fit score from labeled *(profile, listing, label)* examples |
| **Signals** | Text overlap only | Text similarity, skill overlap, career-change accessibility (years asked, degree, training), seniority gap, O*NET distance between the old and new occupations, location and remote |
| **Profiles needed** | One works; more are only for testing | Many varied profiles, **with labels** |
| **Labels needed** | None | Yes, the hardest part (section 4) |
| **Role in the project** | **Baselines**, steps 1–4 of the ladder in `ML_Fit` | Step 6 of the ladder in `ML_Fit`, with skills-based matching (step 5) as one of its inputs |

## 2. Why Option B

1. **Similarity matching works against career changers.** A career changer's résumé is written in their *old* field's language, so Option A ranks old-field jobs highest, which is exactly what they're trying to leave. Option B can learn that transferable skills and openness to newcomers matter more than shared wording.
2. **Fit is more than text.** Years of experience asked, degree or equivalent experience, seniority and occupation distance all decide whether a career changer can get a job. Only a trained model can combine them with learned weights.
3. **It's the project's model.** `ML_Fit` needs a trained, validated, explainable model (SHAP for "why this role"), not a single similarity score; `ML_Rank` combines its fit score with the listing model.
4. **Option A stays as the baseline,** so we can show that Option B actually adds something.

**The ladder in `ML_Fit` (decided 2026-10-10):**

| Step | Method | Role |
|:-:|:--|:--|
| 1 | Keyword overlap | Simplest baseline |
| 2 | TF-IDF cosine | Baseline ML model |
| 3 | BGE embeddings, cosine (small or base model, run locally) | Stronger similarity |
| 4 | RRF hybrid (reciprocal rank fusion of TF-IDF and BGE) | Hybrid similarity |
| 5 | Skills-based matching (both sides mapped to O*NET skills) | Fixes the old-field vocabulary problem |
| 6 | Trained layer (Option B): steps 2–5 scores plus Profiles features, trained on real career moves (logistic regression, gradient boosting, and a small neural network if it wins) | The project's fit model |
| 7 | Stretch: BGE fine-tuned on real career moves (local, small model) | Only if step 6 plateaus |

**Every step is scored the same way:** using only a career changer's pre-switch background, is the role bucket they actually moved into in the top 3 or top 5? Reported per pool and overall, against a title-keyword baseline, along with how often each step suggests the old field. Expect steps 1–4 to favor the old field; steps 5 and 6 exist to fix that.

## 3. Applicant profiles

> **Started first: real career switches from a public resume dataset.** `notebooks/Profiles.ipynb` finds people in the Kaggle structured resume dataset (about 54,000 resumes, MIT license) whose job history shows a move into a computer and mathematical role. Its rules (`src/career_switch.py`) are tuned in review rounds until the candidates are right. These give real pre-switch profiles and real labels without waiting for the advisor. The volunteer profiles below add career changers from the team's own backgrounds, in more detail.
>
> **Result so far:** **2,813 career changers** from the public resume dataset (funnel in [`docs/data.md`](data.md)). The rules were spot-checked on random samples (Profiles 4c). The volunteer profiles below are now a recent test set, not the main source.
>
> **The rules label instead of rejecting:** everyone who reached a computer and mathematical job is kept with labels (origin, retrained, first tech job level, every tech role reached). The role model trains on everyone, with origin as a feature, and its results are reported on career changers (retrained from another field, or changed roles within tech; any job level). A check compares training on everyone with training on career changers only.
>
> **Download** (once, about 136 MB unzipped; no Kaggle login needed). From the project folder:
>
> ```bash
> mkdir -p data/external/resume_structured
> curl -L -o data/external/resume_structured.zip "https://www.kaggle.com/api/v1/datasets/download/suriyaganesh/resume-dataset-structured"
> unzip -o data/external/resume_structured.zip -d data/external/resume_structured && rm data/external/resume_structured.zip
> ```
>
> Or download the zip from the Kaggle page by hand and unzip it into that folder. The notebook finds the CSVs in any subfolder.

### Who we recruit

**People who have already switched** from one background to a different role or field (or are in the middle of it), ideally from the team's backgrounds and into computer and mathematical roles. *Why:* their history gives us real labels. Their profile *before* the switch is the career changer, and the role they moved into is a known good fit (section 4).

### Step 1: consent (before anything else)

Each volunteer signs a short consent form that covers:
- what we collect (job titles, dates, skills, education, a short description of each role)
- what we leave out (name, contact details, photo and employer names)
- that it's used only for this project, stored on project machines only, never uploaded to AI tools, and deleted at the end
- that they can withdraw at any time

Keep the signed forms outside the repo.

### Step 2: the volunteer exports their own LinkedIn data

LinkedIn only lets people export **their own** data, so each volunteer does this themselves. Scraping other people's profiles breaks LinkedIn's terms and our data rules, so we don't do it.

1. On LinkedIn, click **Me → Settings & Privacy → Data privacy → Get a copy of your data**. Menus change, so search the settings for "Get a copy of your data" if this path doesn't match.
2. Choose **"Want something in particular?"** and tick **Positions, Profile, Education, Skills** (and **Certifications** if they have any).
3. Click **Request archive**. LinkedIn emails a download link, usually within minutes; a full archive can take up to 24 hours.
4. The volunteer sends the `.zip` to the **one teammate who collects profiles**, through an approved channel (not email attachments, not personal cloud drives).

**No LinkedIn?** The volunteer fills in `profile_template.csv` instead (`docs/profile_template.csv`: one row per position, skill, degree or certification), using the same fields as section 3, Step 3.

### Step 3: convert to one clean profile per person

The collecting teammate does this locally:

1. Save the zip as `data/profiles/raw/P001.zip` (`P001`, `P002`, … in the order received). The `data/` folder is git-ignored.
2. Read the CSVs (`Positions.csv`, `Skills.csv`, `Education.csv`, `Certifications.csv`). Check the column headers first; LinkedIn has renamed them before.
3. **Remove personal details:** names, email, phone, address, birth date, URLs, photo. Replace employer names with the industry (for example "regional hospital" → "healthcare").
4. **Code every job title to an O*NET occupation** by searching the title on [O*NET OnLine](https://www.onetonline.org/). This connects the profile to the listings' `onet_data`.
5. **Mark the switch:** the date the person moved into the new field, the target role's O*NET code, and the source and target fields.
6. Save one JSON file per person to `data/profiles/clean/P001.json`:

```json
{
  "profile_id": "P001",
  "seed_id": "P001",
  "is_synthetic": false,
  "source_field": "customer service",
  "target_field": "IT support",
  "switched_on": "2024-05",
  "target_onet_code": "15-1232.00",
  "positions": [
    {"title": "Customer Service Representative", "onet_code": "43-4051.00", "industry": "retail",
     "start": "2018-03", "end": "2024-04", "description": "Resolved customer issues by phone and chat ..."}
  ],
  "skills": ["troubleshooting", "ticketing systems", "customer communication"],
  "education": [{"degree": "associate", "field": "business"}],
  "certifications": ["CompTIA A+"]
}
```

*(Example values are made up.)*

7. **Cut the profile at the switch:** keep only positions, skills and certifications from *before* `switched_on`. Anything from after the switch leaks the answer.
8. Delete the raw zip once the clean JSON is checked. Log every remaining copy for the end-of-project deletion.

### Step 4: synthetic profiles, only to balance small roles

**The 1,000-profile plan is dropped (2026-10-09):** it assumed about 50 real profiles, and there are now 2,813 real career changers. Roles are uneven (data and analytics is small: 237 people, only 19 of them retrained into it), so:

- **The compass and every reported share use real people only.** "30% of developers who changed roles moved into data" is a fact; balancing would make every role look equally common.
- **The model handles imbalance in training:** class weights first (`class_weight="balanced"`), with per-role scores so a small role isn't hidden in an average.
- **Only if a small role is still predicted poorly:** add synthetic people *of that role*, made by perturbing real ones with the structured edits below, **inside each training fold only**, kept in the same fold as their seed. Never in the test set, the compass or any reported number.
- **The model's probabilities are corrected back to the real role mix** before role discovery shows them; otherwise a model trained on balanced data over-predicts small roles.

Each synthetic person is a variation of a real **seed**, with one or two **structured edits**, each recorded:

| Edit | Example | Rule that keeps it realistic |
|:--|:--|:--|
| Change years in a role | 3 years → 6 years | Keep dates consistent; no overlapping jobs |
| Add or drop skills | add "SQL"; drop "scheduling" | New skills come from the O*NET skill list of an occupation the person held |
| Swap a past role for a nearby one | retail associate → cashier | Use O*NET's related occupations |
| Change education | none → associate; add a certificate | Only levels common in the source field |
| Change the target field | IT support → data analyst | Only O*NET-related targets |

Each synthetic profile keeps its seed's `seed_id`, sets `"is_synthetic": true`, and lists its edits in an `"edits"` field. **Don't use an LLM to write profiles.** LLM-written profiles share one writing style, and the model learns the style instead of the fit. The generator script goes in `src/` with a fixed `random_state`.

## 4. Labels

A profile is only useful for training if its *(profile, listing)* pairs have labels.

| Source | Label | Use |
|:--|:--|:--|
| **Each real person's switch** | Listings in the target role they moved into = good fit | Training and **evaluation**: these are real outcomes |
| **Weak rules** | Fit from O*NET occupation distance, skill overlap and the career-change flags | Training only. If we evaluated on these, the model would just be graded on our own rules |
| **Team ratings** (agreed) | Teammates rate about 30 listings per real profile: good / maybe / no | Evaluation. Only teammates see listing text; volunteers never see PredictLeads data |

A synthetic person (only for small roles, inside training folds) takes its seed's role label, adjusted for its edits.

## 5. Validation rules

- **Group folds by `seed_id`** (`GroupKFold`): a real person and any synthetic variations of them always sit in the same fold. Otherwise the model sees near-copies of the test profiles during training.
- **Report final results on real people only:** the career changers from `Profiles.ipynb`, and volunteer profiles as a later test set. Synthetic profiles are only for small roles in training and for stress tests.
- **Compare against Option A** and a title keyword search: NDCG@10 and precision@10, as mean ± std across applicants.
- **Role discovery:** for each real switcher, is the role they actually moved into among the top 3 and top 5 suggested roles? Compare with a title-keyword baseline.
- **Rate only visible listings:** listings hidden as unreachable are left out of labels and metrics, because the user never sees them.
- **Stress tests on synthetic profiles:** does the ranking move sensibly when someone gains 3 years of experience or adds a certificate?

## 6. Who does what

| Task | Owner | Depends on |
|:--|:--|:--|
| Draft the consent form; get the advisor's approval | TBD | Advisor's approval to collect profiles |
| Recruit 50 career changers from the chosen source backgrounds | TBD (everyone recruits) | Consent form |
| Collect zips and convert them to clean JSON (section 3, Step 3) | One teammate only | Profiles arriving |
| Code job titles to O*NET | TBD | Clean JSON |
| Write the synthetic generator in `src/` | TBD | O*NET related occupations and skills downloaded |
| Write the label rules; collect team ratings | TBD | Profiles and EDA_SDA_2's scope table |

**Never** paste profiles, listing text or any of this data into AI tools (Claude, ChatGPT, Copilot). Use made-up examples like the one above when asking for coding help.
