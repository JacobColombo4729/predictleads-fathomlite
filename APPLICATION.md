# Application

**Status:** Draft for discussion · **Last updated:** 2026-10-09

> A web app for career changers. It turns the analysis (see [`PROJECT_OUTLINE.md`](PROJECT_OUTLINE.md)) into a tool a job seeker uses day to day.
> Items marked **TBD** need a team decision. Follow the [Data & Security Rules](README.md#data--security-rules).

---

## 1. Vision

A career changer has limited time, a limited number of applications, and a background most listings in their new field don't ask for. The app shows them **where an application is most worth spending**: live postings in the field they're moving into, in a searchable list, ranked by a priority model we tune ourselves, with the reasons shown. Every listing answers two questions: **can I realistically get it** (how open it is to someone new to the field, and which of their skills carry over) and **is it worth applying now**.

**The user** is a career changer (anyone switching from their current background to a different role or field) moving into computer and mathematical roles, from one of the team's backgrounds: a teammate with their own profile, or a synthetic career changer. The app does two things: **role discovery** (which roles fit them) and the **priority queue** (which listings in those roles to apply to first). Listings they can't get are hidden.

The priority model isn't guesswork. It's built from the full analysis, starting with the EDA in `notebooks/EDA_SDA_1.ipynb`. Every number the app shows (openness to career changers, apply-by date, chance still open, transferable fit, competition, pay) comes from a target we studied and validated there.

The core of the app is the **priority queue**: listings ranked for this career changer. The timing advice from the analysis sits around it:

| From the analysis | In the app |
|---|---|
| Prioritization model | The priority queue: the list sorted by the priority score |
| Career-change accessibility | Listings a career changer can't get are hidden; the rest show what they ask of a newcomer: years asked, degree or equivalent experience, training offered |
| Still open / apply-by | An estimated apply-by date on every listing, and an apply-by filter |
| Source and arrival cadence | Check schedule: which sources and companies to watch, and how often; alerts for new listings |
| Seasonal context | Market panel: is this a busy or quiet month for the field, compared with a normal year |

---

## 2. Screens and features

### List view (home)

- **Live search:** the list narrows with every letter typed. It matches titles, companies and, optionally, description text.
- **Sort:** priority (default), openness to career changers, apply-by date, transferable fit, pay, freshness.
- **Filters:** they combine, and each one shows how many postings match:
  - **Priority:** minimum score, or top N
  - **Apply-by:** closing within N days, or before a date
  - **Hidden automatically:** listings a career changer can't get (manager level and above, too many years asked, or a degree or license this user lacks). A note shows how many were hidden, and why
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
- Each suggested role shows the skills that carry over, the main gaps, how many reachable listings are open now, and median pay.
- Picking roles sets what the priority queue ranks.

### Background profile (for transferable fit)

- The user describes where they're coming from: a résumé (PDF or Word), pasted text, or their LinkedIn data export (the Profile, Positions, Education and Skills files).
- They pick the field they want to move into, or let the app suggest related fields from O*NET's related occupations.
- The profile is compared with postings using TF-IDF or local embeddings, with O*NET skills connecting the old and new occupations, giving a transferable-fit score for every posting.
- **Privacy:** the résumé stays local. Personal details are never stored or logged beyond the session unless the user chooses to save them.

### Market panel and check schedule

- **Market:** this month's new-posting volume for the field vs. its usual level, with the seasonal outlook.
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
| **Pay** | Annual salary, where disclosed | Cleaned salary data, compared with market benchmarks |
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
| `market` | Seasonal outlook; wage benchmarks; market tightness | External public data |
| `tracker` | Star, set status, list starred postings, reminders | The user's own data |

### Agents

| Agent | Job | Servers it uses |
|---|---|---|
| **Background** | Reads the career changer's profile and extracts past roles, skills and target fields | `fit` |
| **Scout** | Follows the check schedule and finds new postings that match the user | `postings`, `timing` |
| **Ranker** | Scores and ranks postings into the priority queue | `priority`, `timing`, `fit` |
| **Coach** | Explains the queue and the strategy in plain language; turns questions into filters ("what should I apply to this week?") | All, read-only |
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
| BLS JOLTS (job openings, hires, quits) | National hiring seasonality and openings per hire, a measure of competition | Market panel, competition, and a check on our own seasonal findings |
| BLS Occupational Employment and Wage Statistics | Wage benchmarks by occupation and area | Pay vs. market in the priority score |
| O*NET skills, abilities and related occupations | The skills each occupation needs, and which occupations are close | Transferable fit; suggesting target fields from a background; gap explanations |
| BLS unemployment by occupation or industry | How many job seekers compete for the same roles | Competition |
| Public holiday and fiscal calendars | Explains seasonal dips and spikes | Market panel |

---

## 6. Build in stages

Each stage is usable on its own, so we always have something to demo.

| Stage | What | Depends on |
|---|---|---|
| **1. MVP** | List view with live search, sort and filters; detail view with apply link; stars and Starred tab; unreachable listings hidden, with what each listing asks of a newcomer; apply-by, chance still open, pay, freshness | Still open / apply-by and the EDA's chosen factors |
| **2. Transferable fit and role discovery** | Background profile; suggested roles; transferable-fit score and filter; skills that carry over, and gaps | TF-IDF or local embeddings; O*NET skills |
| **3. Priority model** | Tuned priority score with adjustable weights | Full analysis |
| **4. Agents** | MCP servers, then the agents, starting with Ranker, Coach and Reminder | Stages 1–3; the advisors' answer |
| **5. External data** | Market panel, wage benchmarks, market tightness | Stage 1 |

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
- **Queue fields:** is anything missing, such as a skills-gap column, salary vs. market benchmark, or company information?
- **Advisors:** may an agent backed by an outside LLM see posting text, or only aggregates and model outputs?
- **External data:** which datasets are allowed, and do we need permission to combine them with PredictLeads data?
- **Hosting:** does the app need to be hosted, or is a local demo enough?
- **Front end:** is Streamlit enough for live search and full-screen detail views, or do we need React?
