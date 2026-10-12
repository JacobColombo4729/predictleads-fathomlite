# Volunteer Career-Changer Profiles

**Status: on hold** until the project advisor approves collecting profiles (task "Applicant profiles" in [`PRIORITY_QUEUE.md`](../PRIORITY_QUEUE.md), Blocked). Nothing has been collected.

## What this is

A plan to test the fit model on **real career changers we recruit ourselves**: people who have already moved from another field into a tech role (or are in the middle of it), ideally from the team's own backgrounds. Their profile **before** the switch is the input, and the role they moved into is a known good answer, so they make a recent, detailed **test set**.

The model's main training data doesn't depend on this: [`Profiles.ipynb`](../notebooks/Profiles.ipynb) already finds 2,813 career changers in a public resume dataset. Volunteer profiles only add a recent check on top. Final results are reported on real people only.

## How a volunteer provides a profile

1. **Consent first.** Each volunteer signs a short form covering what we collect (job titles, dates, skills, education, a short description of each role), what we leave out (name, contact details, photo, employer names), that it's used only for this project, kept on project machines only, never uploaded to AI tools and deleted at the end, and that they can withdraw at any time. Signed forms stay outside the repo.
2. **Either export their own LinkedIn data:** on LinkedIn, **Me → Settings & Privacy → Data privacy → Get a copy of your data** (search the settings for "Get a copy of your data" if the menus have moved), choose **"Want something in particular?"**, tick **Positions, Profile, Education, Skills** (and **Certifications**), and request the archive. LinkedIn emails a download link. Only people's **own** data: scraping other profiles breaks LinkedIn's terms and our data rules.
3. **Or, without LinkedIn, fill in [`volunteer_profile_template.csv`](volunteer_profile_template.csv)** (in this folder): one row per position, skill, degree or certification, the same information the LinkedIn export holds. It's a blank form; the filled-in copy is personal data and never goes back into the repo.
4. **Send it to the one teammate who collects profiles,** through an approved channel (not email attachments, not personal cloud drives).

## What the collecting teammate does (locally)

1. Save each submission as `data/profiles/raw/P001.zip` or `.csv` (`P001`, `P002`, … in the order received; `data/` is git-ignored).
2. **Remove personal details:** names, email, phone, address, birth date, URLs, photo; replace employer names with the industry ("regional hospital" → "healthcare").
3. **Code every job title to an O*NET occupation** on [O*NET OnLine](https://www.onetonline.org/), so profiles connect to the listings' `onet_data`.
4. **Mark the switch:** the date the person moved into the new field and the target role's O*NET code.
5. **Cut the profile at the switch:** keep only positions, skills and certifications from before it; anything later leaks the answer.
6. Save one clean JSON per person in `data/profiles/clean/`, delete the raw file once checked, and log every copy for the end-of-project deletion.

## How they're used

- As a **test set** for role discovery (is the role the person actually moved into among the top 3 and top 5 suggestions?) and for the listing ranking, next to the public-resume career changers.
- Folds are grouped by person; results are reported on real people only.

## Rules

- **Never** paste a profile, or any of this data, into AI tools (Claude, ChatGPT, Copilot); use made-up examples when asking for coding help.
- Profiles stay under `data/` and are never committed.

*The full original plan (including synthetic profiles to balance small roles) was `docs/fit_analysis.md`, removed 2026-10-11; it's in git history.*
