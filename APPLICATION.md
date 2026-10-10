# Application

**Status:** Draft for discussion · **Last updated:** 2026-10-09

> A web app for career changers. It turns the analysis (see [`PROJECT_OUTLINE.md`](PROJECT_OUTLINE.md)) into a tool a job seeker uses day to day.
> Items marked **TBD** need a team decision. Follow the [Data & Security Rules](README.md#data--security-rules).

---

## 1. Vision

A career changer has limited time, a limited number of applications, and a background most listings in their new field don't ask for. The app shows them **where an application is most worth spending**: live postings in the field they're moving into, in a searchable list, ranked by a priority model we tune ourselves, with the reasons shown. Every listing answers two questions: **can I realistically get it** (how open it is to someone new to the field, and which of their skills carry over) and **is it worth applying now**.

**The user** is a career changer (anyone switching from their current background to a different role or field) moving into computer and mathematical roles, from one of the team's backgrounds, or a teammate with their own profile. The app does two things, with the **retraining compass** at the center: **role discovery** (which roles fit them, which paths lead there, and what to build next) and the **priority queue** (which listings in those roles to apply to first). Listings they can't get are hidden. Someone already in tech or math gets the same tools for **moving up the career ladder** into a different role.

The priority model isn't guesswork. It's built from the full analysis, starting with the EDA in `notebooks/EDA_SDA_1.ipynb`. Every number the app shows (openness to career changers, apply-by date, chance still open, transferable fit, competition, pay) comes from a target we studied and validated there.

The core of the app is the **priority queue**: listings ranked for this career changer. The timing advice from the analysis sits around it:

| From the analysis | In the app |
|---|---|
| Prioritization model | The priority queue: the list sorted by the priority score |
| Career-change accessibility | Listings a career changer can't get are hidden; the rest show what they ask of a newcomer: years asked, degree or equivalent experience, training offered |
| Still open / apply-by | An estimated apply-by date on every listing, and an apply-by filter |
| Source and arrival cadence | Check schedule: which sources and companies to watch, and how often; alerts for new listings |
| Job growth expectations | Expected growth of each role (time frame to be decided), shown in role discovery and the compass |
| Listing activity over time | A date picker: the queue as it stood on any date in the data |

---

## 2. Screens and features

### List view (home)

- **Date picker ("as of"):** shows the queue as it stood on a chosen date, for example June 1 vs. August 1, 2026. See [Pick a date](#pick-a-date-as-of-view).
- **Live search:** the list narrows with every letter typed. It matches titles, companies and, optionally, description text.
- **Sort:** priority (default), openness to career changers, apply-by date, transferable fit, pay, freshness.
- **Filters:** they combine, and each one shows how many postings match:
  - **Priority:** minimum score, or top N
  - **Apply-by:** closing within N days, or before a date
  - **Hidden automatically:** listings a career changer can't get (too many years asked, a degree or license this user lacks, or manager-level roles if the user has no management experience). A note shows how many were hidden, and why
  - **Training offered:** only listings that offer training or welcome career changers
  - **Transferable fit:** minimum score (needs the user's background profile)
  - **Pay:** minimum salary; salary disclosed only
  - **Freshness:** first seen within N days
  - **Competition:** low, medium or high
  - **Source:** hiring platform or company site; specific platforms
  - **Location and remote**
  - **Seniority and contract type**
- **Each row shows:** title, company, priority score, an "open to career changers" badge, apply-by date, transferable fit, pay and a star.

### Starred postings

- **Star** any posting to save it.
- A **Starred** tab shows only starred postings. Every search, sort and filter works on that list too, so the user can rank their own shortlist.
- Starred postings also act as a tracker: saved, applied, interviewing, rejected, or skipped.
- **Reminders** when a starred posting's apply-by date is close.

### Detail view

Opens full screen on tap or click:
- Title, company (name, domain, location), pay, source platform
- **Apply-by date** with its estimated chance still open, and a short explanation of what it's based on
- **What it asks of a newcomer:** years of experience, degree or equivalent experience, licenses, training offered
- **Transferable fit** with the skills that carry over from the user's old field, and gaps (requirements their profile doesn't show yet)
- **Competition** signals: reposted, company hiring at volume, similar postings open
- **Priority score** broken into the parts that make it up
- The full posting text
- **Apply** button: a link to the application portal (the posting's original URL)
- Star, and set tracker status

### Role discovery ("Roles for you")

- From the background profile, suggests the computer and mathematical roles (O*NET occupations) their skills transfer to.
- Learned from real career switches: people in a public resume dataset who retrained into computer and mathematical roles (`notebooks/Profiles.ipynb`).
- Each suggested role shows the skills that carry over, the main gaps, how many reachable listings are open now, and median pay.
- Picking roles sets what the priority queue ranks.

### Retraining compass

- **Two uses:** changing careers (retraining into a new field) and **moving up** (training that helps someone reach senior or manager roles in their own field, for example a master's or a certification before a senior or lead role).
- Starts from the user's **prior career**: where people from that career landed (Profiles 4a.2b).
- For the user's background, lists the **training options** people like them took (degrees, certificates, bootcamps), grouped by field.
- For each one: the **tech roles it led to** (from real career switches, with how many people), and for each role its **2026 openings, median pay, how fast listings close and its expected job growth**.
- Picking a role sets what the priority queue ranks. Worded as "people like you got there through this training", never "this training gets you hired" or "your odds are X%": the resumes show only people who made a switch.
- **Every path shows how many people it rests on** ("based on N people"); paths under about 10 people are marked as thin.
- **Job growth carries a caveat:** it's for the whole occupation, so it sits next to the role's entry-level share from our listings.
- **Questions about which path to take**, answered from the resume data (the Coach agent explains them):
  - Should I retrain for the role directly, or take a first tech role and move from there? (Data and analytics: 216 career changers moved in from another tech role, only 22 retrained straight into it.)
  - Which training led people like me to this role, and how many people is that based on?
  - How long did it take? (People who retrained after working took a median of 14 months from their last old-field job to their first tech job.)
- **Compass ideas** (from the team meeting; to try, not decided). Every number is labelled an estimate, worded as "people like you" and "expected", never a promise, and the résumé stays local:
  - **"Based on your current résumé, we think you should build these skills":** skills that listings in the suggested roles ask for and the résumé doesn't show (listing text from EDA_SDA_3, O*NET skills), most-asked first.
  - **"Here's what your career options could look like down the line":** the roles people with a similar background reached, and how long it took them (Profiles), with each role's 2026 openings.
  - **"Here's your salary increase":** median 2026 listing pay in each option vs. the user's current role (from listings in their current occupation, or a salary they enter).
  - **"Here's the job growth we expect in your career":** expected employment growth (%) for each option.
  - **Stepping-stone roles:** a role that bridges two functions on the way to a bigger move (for example engineer → pre-sales engineer → director, picking up business experience). The resume rules already record the level at the switch (senior, manager or above), so the compass can show which roles people passed through.

### Moving across and up (already in tech or math)

- For a user already in a computer and mathematical role: the **roles people in that role moved into** (1,848 people in the resumes changed role buckets within tech after at least 12 months), with how many people, and each role's 2026 openings, pay, closing speed and expected growth.
- Examples from the data: IT support → networks and systems or security; developers → databases or data and analytics.
- Each move shows whether people moved sideways or up (into a senior or manager role), from Profiles 4a.3.
- Picking a role sets what the priority queue ranks, like role discovery.

### Pick a date (as-of view)

- The user picks a date; the queue shows the listings **live on that day** and ranks them with **only what was known that day**. Comparing June with August shows how the best options change.
- **What the dates mean:** `first_seen_at` is the first day PredictLeads' crawler found a listing (a median of 1.5 days after the employer posted it); `last_seen_at` is the last day the crawler still found it. A listing is live on a date when it was first seen on or before it and last seen on or after it.
- **No hindsight in the scores:** listing age is counted up to the chosen date; reposts and company hiring volume only from listings first seen before it; and `last_seen_at` only decides which listings were live, never a score.
- **No hindsight in the model:** the model that scores a date is trained only on what was known by that date; a listing still open on that date counts as still open in training, even though we later know when it closed. The simple version for the demo offers only dates in the test period (September), which the main model never learned from; any earlier date needs a model retrained as of that date.
- **A built-in check:** for past dates we know which listings actually closed and when, so the app can show how good its apply-by estimates were.
- Dates before June miss long-open listings first seen before 2026-03-30 (the data's six-month window).

### Background profile (for transferable fit)

- The user describes where they're coming from: a résumé (PDF or Word), pasted text, or their LinkedIn data export (the Profile, Positions, Education and Skills files).
- They pick the field they want to move into, or let the app suggest related fields from O*NET's related occupations.
- The profile is compared with postings using TF-IDF or local embeddings, with O*NET skills connecting the old and new occupations, giving a transferable-fit score for every posting.
- **Privacy:** the résumé stays local. Personal details are never stored or logged beyond the session unless the user chooses to save them.

### Market panel and check schedule

- **Market:** this month's new-posting volume for the field, and each role's expected job growth.
- **Check schedule:** "check Greenhouse and these 5 companies every 3 days," based on how fast new listings arrive and close.

---

## 3. The priority model

The priority score ranks postings by how much an application is worth. It combines:

| Factor | Meaning | Source |
|---|---|---|
| **Chance still open** | Probability the posting is still open today | Survival model |
| **Apply-by date (estimated)** | The date by which the posting is likely to have closed | `first_seen_at` + the predicted window for postings like it. Postings rarely list a real deadline, so this is always labelled an estimate. |
| **Career-change accessibility** | How open the posting is to someone new to the field | Seniority and listing text: years required, degree or equivalent experience, training, career-changer wording |
| **Transferable fit** | How well the skills from the user's old field match the posting | TF-IDF or local embeddings, with O*NET skills |
| **Competition** | Lower is better | Reposts, company hiring volume, similar open postings, external market tightness |
| **Pay** | Annual salary, where disclosed | Cleaned salary data, compared with listings in the same role |
| **Freshness** | Days since first seen | `first_seen_at` |

**How it's tuned:** the analysis decides which factors carry real signal and how strongly. The weights start from those findings and are checked against a newest-first baseline on a later time period. Users can adjust the weights with sliders, for example "reachability matters more to me than pay."

Every score shows its parts, so a user can see why a posting ranks where it does.

---

## 4. Multi-agent MCP design

### Why it fits

Each MCP server wraps one capability we build anyway: postings, timing, fit, market data, the user's tracker. The agents then do real jobs a job seeker would otherwise do by hand. This has four benefits:
- **Data boundaries:** each server enforces what leaves it, which is how we keep confidential posting text away from outside AI services.
- **Natural-language control:** "show remote analyst roles closing this week with fit above 70%" becomes filters on the list view.
- **Background work:** agents watch for new postings and approaching deadlines while the user is away.
- **Swappable parts:** a better fit model or a new data source plugs in without changing the app.

### MCP servers

| Server | Tools (examples) | Data |
|---|---|---|
| `postings` | Search and filter live postings; get one posting's features | Our cleaned postings, stored locally |
| `timing` | Chance still open; apply-by date; check schedule | Our survival and arrival models |
| `fit` | Suggest roles from a background profile; score it against postings; explain which skills transfer; list gaps | Local TF-IDF or embeddings, O*NET skills |
| `priority` | Score and rank postings with the tuned model and the user's weights | Outputs of `timing`, `fit` and competition signals |
| `market` | Job growth expectations by role; listing volume by month | Job growth expectations; our listings |
| `tracker` | Star, set status, list starred postings, reminders | The user's own data |

### Agents

| Agent | Job | Servers it uses |
|---|---|---|
| **Background** | Reads the career changer's profile and extracts past roles, skills and target fields | `fit` |
| **Scout** | Follows the check schedule and finds new postings that match the user | `postings`, `timing` |
| **Ranker** | Scores and ranks postings into the priority queue | `priority`, `timing`, `fit` |
| **Coach** | Explains the queue, the retraining compass and the strategy in plain language; answers questions about which path to take; turns questions into filters ("what should I apply to this week?") | All, read-only |
| **Bridge** | Compares the career changer's profile with a posting's requirements, and suggests what to emphasize from their old field and what to learn | `fit` |
| **Reminder** | Flags starred postings whose apply-by date is close, and new high-priority matches | `tracker`, `timing` |
| **Orchestrator** | Decides which agent runs when, and combines their answers | All |

### Data rule for the agents

Under the advisors' rules, no confidential data may go to outside AI services. Until the advisors approve otherwise in writing:
- An agent backed by an outside LLM only sees **non-confidential data**: the user's own résumé (with their consent), aggregate statistics, public external data, and model outputs (scores, dates, counts). Never raw posting text.
- Anything that reads posting text (Bridge, the detail view, fit) runs **locally**: our own models, or a local LLM, for example through Ollama.
- Demos that show an LLM reading postings use **synthetic postings**.

### Bonus: works with any MCP client

Because the tools are standard MCP servers, they can also be used outside our app, for example from Claude Desktop. A user could ask their own assistant "what are my top 5 postings this week?" That's a strong demo of the multi-agent design.

---

## 5. External data (candidates)

All public. Each would need its terms of use checked.

| Dataset | What it adds | Used in |
|---|---|---|
| Job growth expectations (official occupational employment projections, e.g. BLS) | Expected employment growth by occupation; published for 10 years and converted to the time frame we choose at the same yearly rate | Role discovery, retraining compass, market panel. Joins directly on occupation code |
| O*NET skills, abilities and related occupations | The skills each occupation needs, and which occupations are close | Transferable fit; suggesting target fields from a background; skills to build (compass) |

---

## 6. Build in stages

Each stage is usable on its own, so we always have something to demo.

| Stage | What | Depends on |
|---|---|---|
| **1. MVP** | List view with live search, sort and filters; date picker; detail view with apply link; stars and Starred tab; unreachable listings hidden, with what each listing asks of a newcomer; apply-by, chance still open, pay, freshness | Still open / apply-by and the EDA's chosen factors |
| **2. Transferable fit and role discovery** | Background profile; suggested roles; retraining compass; compass ideas; moving across and up within tech; job growth expectations; transferable-fit score and filter; skills that carry over, and gaps | TF-IDF or local embeddings; O*NET skills; job growth expectations |
| **3. Priority model** | Tuned priority score with adjustable weights | Full analysis |
| **4. Agents** | MCP servers, then the agents, starting with Ranker, Coach and Reminder | Stages 1–3; the advisors' answer |
| **5. Market panel** | Listing volume by month, with each role's expected growth | Stage 1 |

---

## 7. Logistics

| Topic | Draft | Status |
|---|---|---|
| Front end | Streamlit for speed (Python only); a React front end if we need richer interaction such as instant search-as-you-type and full-screen detail views | TBD |
| Back end | Python; FastAPI for the app; MCP servers built with the official `mcp` Python SDK | TBD |
| Storage | Local SQLite or Parquet for postings and scores; SQLite for stars and the tracker | TBD |
| LLM for agents | Outside API on non-confidential data only, or a local model | TBD, needs the advisors' answer |
| Hosting | Local demo only, since the data can't leave approved locations | TBD |
| Timeline | MVP at the end of week 2; fit, priority model and agents in week 3 | TBD |
| Owner | | TBD |

---

## 8. Open questions

- **Course requirements:** what's required for the web app and the multi-agent part (rubric, must-haves), or is it open-ended?
- **Agent ambition:** are agents the centerpiece of the app, or a layer on top of a solid queue? This decides how much of week 3 goes to them.
- **Team split:** who is best suited to own the app (front end, MCP servers), so the analysis isn't starved?
- **Queue fields:** is anything missing, such as a skills-gap column or company information?
- **Time frames:** how far ahead the compass looks for career options and job growth (not decided yet).
- **Advisors:** may an agent backed by an outside LLM see posting text, or only aggregates and model outputs?
- **External data:** which datasets are allowed, and do we need permission to combine them with PredictLeads data?
- **Hosting:** does the app need to be hosted, or is a local demo enough?
- **Front end:** is Streamlit enough for live search and full-screen detail views, or do we need React?
