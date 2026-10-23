# How it works, and how we know

**For career changers moving into tech:** which roles your background transfers to, which listings you can realistically get, and which to apply to first. Built on PredictLeads' 2026 job listings and 2,813 real career switches from public résumés. Start here; everything else is reference (links at the end).

## Three numbers

| | Result | Against | Where |
|:--|:--|:--|:--|
| **Roles** | For **81.7%** [78.5%, 84.7%] of 563 real career changers the model never saw, the role they actually moved into is in our top 3 | 63.2% for suggesting the most common roles | ML_Fit 6b |
| **Listings** | **Half** (0.500) of each career changer's top 10 is relevant: still open 14 days later, not a ghost, and in the role they moved into | 0.031 for newest first; 0.071 for a title search | ML_Rank 8e, test month |
| **Timing** | **29%** of listings are gone within a week (median 24 days) | | EDA_SDA_1 Q1b |

The robust part of the listings result is the role: 50.4% of our top 10 is in the role the person moved into, against 23.8% for the best possible title search. (Newest first and the title search give everyone the same 10 listings, so their other shares rest on those 10.)

## One path: résumé → role fit → candidate pool → queue

| Box | What it does | How we know it works | What could go wrong |
|:--|:--|:--|:--|
| **Résumé** | Reads a résumé on the user's computer and turns it into a career timeline with the same rules used on the 2,813 career changers (`src/fit_serving.py`) | It gives exactly Profiles' features for people who switched at a known month (tests) | A scrambled résumé gives a scrambled timeline; undated entries can't be placed |
| **Role fit** | A trained model predicts which role someone moves into from their background, with the reasons | The 81.7% above, on 563 people it never saw | Only 8 roles have a fit model; growth is not in it yet |
| **Candidate pool** | The best text matches overall, plus the best within the person's top roles; listings they can't get are left out | The top roles add about 210 listings the text match alone misses (fictional people) | The text match decides which listings of a role are considered |
| **Queue** | Ranks the pool by a weighted sum: fit, chance still open, how open the listing is to career changers, and pay | The 0.500 above, on a month the weights never saw | Listings outside the 8 roles get a neutral fit; the weights come from one validation month |

**The score on each listing** is that weighted sum: each factor is rated 0–100 and multiplied by its tuned weight (fit 0.3, still open 0.2, accessibility 0.1, pay 0, so fit is half the score).

**The app** runs this whole path (`combined_demo/`, version 2): **68.2%** [64.5%, 72.1%] of the same 563 career changers have their real role among the roles of its top 20 listings, against 69.4% for the queue without role fit (`mlrank`), 67.9% for the text match alone (`rrf`) and 63.2% for the baseline (Evaluate). **On this answer key the three rankers are not separated from each other.** Version 1 scored 76.4% by taking roles from the role-fit model alone; version 2 blends in the text match so that a real résumé leads with the right role and a stated desired role comes first — neither of which this answer key can show, because these backgrounds are the role-fit model's own inputs and carry no desired role. The claims that stand on held-out people are the role result (81.7%) and the queue (0.500).

## How we know: graded on real outcomes

- **The answer key is real:** each of the 2,813 career changers has the role they actually moved into. A method sees only their background before the switch. No labelling, no AI grader.
- **Always next to a baseline:** every number above is shown against the simplest reasonable guess.
- **Tested on people and months it never saw ("held out"):** 563 of the 2,813 career changers, and the last month of listings, were set aside before anything was built. The models were built and tuned on the rest, then scored on the set-aside part once, at the end, so the scores show how they do on new people, not how well they memorised the old ones.
- **No peeking at the future:** listing features use only what's known when a listing first appears; listings still open when the data ends are counted as unfinished, not dropped.
- **Evidence only:** models are built only from the listings and public résumés, never the team's own backgrounds.

The full rules (leakage, confounders, hiding): [`PROJECT.md`](PROJECT.md), How we keep it honest.

## What we don't claim

- **Not hires:** we see listings open and close, not who got the job.
- **Not causation:** "people like you got there through this training", never "this training gets you hired".
- **Not success rates:** the résumés show people who switched, not those who tried and didn't.

## Built once, runs on a laptop

No LLM is trained. The heavy step is encoding the 193,043 listings the app serves, done once on a GPU; the role-fit model trains in minutes on a laptop, and the app runs on one. Details: [`PROJECT.md`](PROJECT.md), Built once, runs on a laptop.

## Where to go next

- **The project:** [`PROJECT.md`](PROJECT.md) (data, findings, the features the models use, how we keep it honest, limits) and [`../DIFFERENTIATORS.md`](../DIFFERENTIATORS.md) (how it stands out, how we pitch it).
- **The analysis:** the notebooks in order, EDA_SDA_1–4 → TimeSeries → Profiles → ML_Fit → ML_Rank → Evaluate (`notebooks/`, readable exports in `docs/`).
- **The app:** [`../app/DESIGN.md`](../app/DESIGN.md); the combined model's design: [`../app/docs/combined-model-plan.md`](../app/docs/combined-model-plan.md).
- **Running it:** [`../README.md`](../README.md) (setup; the combined demo; the hand-offs zip, step 9).
- **What's left before the pitch, and what's cut:** [`ANALYSIS.md`](ANALYSIS.md).
