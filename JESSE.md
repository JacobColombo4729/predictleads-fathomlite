# Jesse's changes

> What Jesse changed, by category, so Jacob can catch up at a glance. Categories and the lines in each are ranked most important first. Each line is the net result (a file moved and later deleted just reads "deleted"), with the date and anything Jacob needs to do. Details are in the commits (`git log`).

## Machine learning

- **Rebuilt your whole pipeline from the repo on a Windows PC, and it reproduced your numbers exactly** (branch `staging`, not merged to `main`). Step 1 (`stream_full`), then EDA_SDA_1, Profiles, EDA_SDA_2 to 4, TimeSeries, ML_Fit and ML_Rank, then `combined.corpus build` and `encode`, then `combined_demo`. ML_Fit's test hit rates match the write-up to the digit and the interval: trained layer 81.7% [78.5%, 84.7%] at top 3 and 96.3% [94.7%, 97.7%] at top 5, Profiles features alone 75.8% / 94.5%, RRF 70.0% / 86.7%, baseline 63.2% / 87.0%. The combined corpus came out at 193,043 listings. (2026-10-17)
- **Role-family classifier fixed: its 94.7% was label leakage** (now on `jesse` and `staging`). The model read the O*NET occupation name inside the saved BGE text, and the labels come from that name. Leak-free accuracy is about 72% (TF-IDF on titles; BGE title embeddings 70.5%; majority class 30%), over 3 company-held-out splits. `app/code/role_family_classifier.py` now embeds titles only (cached in `data/matching/role_family_classifier_cache/`), reports mean over splits, has an optional `--leak-control`, and no longer predicts families for resumes (unvalidated). New guard tests: `app/tests/test_role_family_classifier.py`. **Jacob:** do not quote 94.7%. The Compass still imports `FAMILY_OF_OCCUPATION` unchanged. (2026-10-13)

## App

- **Settings page with an Admin mode** (branch `feature/admin-ml-notes`, not merged). A gear in the side navigation opens Settings; the Admin switch (saved in the browser only, off by default) adds a collapsed "How this works" box beside each machine-learning part: the Discover ranker (with its own wording for the combined ranker and the ML_Rank queue) and 👍/👎 feedback; six boxes in the Compass (the analysis, map and scope, stretch and suggestions, requirement coverage, the GenAI skills gap analysis, next steps); Signals, Ask the market and company research; Magic fill (wording follows the local or OpenAI reader). Each box lists the approach, why we chose it, the data it uses, how it was checked and where it is weak. Frontend only: `lib/admin.tsx`, `lib/mlNotes.ts` (the text, kept in line with `app/docs/ml-inventory.md`), `components/MlNote.tsx`, `pages/SettingsPage.tsx`, `styles/admin.css`. **Jacob:** if a model's numbers change, update `lib/mlNotes.ts` too. (2026-10-17)

## Setup and launchers

- **The three Windows launchers broke when the folder name has a space** (`combined_demo/run.ps1`, `app/run_demo.ps1`, `app/run_jacob.ps1`). Inside `Start-Process ... -Command`, PowerShell strips double quotes, so `C:\Users\me\Professional Projects\...` was split at the space and the frontend (and the combined backend) window failed. Paths in those `-Command` strings are now in single quotes. Tested on a path with a space. **Jacob:** nothing to do on a Mac. (2026-10-17)
- **The README's notebook order was wrong:** EDA_SDA_2 reads `data/profiles/modeling_table.pkl`, which Profiles writes, so the order is EDA_SDA_1, Profiles, EDA_SDA_2, EDA_SDA_3, EDA_SDA_4, TimeSeries, ML_Fit, ML_Rank (README step 9 and `SETUP.md` now say so). `SETUP.md` also lists the spaCy model download (EDA_SDA_3 and ML_Fit need it; the README already said so in its install section). (2026-10-17)
- **GPU for the combined encode:** `requirements.txt` pins `torch==2.14.1`, the CPU-only build on Windows, so encoding the 193,043 combined listings ran at about 4 per second (about 12 hours projected). With the CUDA build of the same release (`cu130`, RTX 3070) it took about 45 minutes in all; the encode resumes from saved chunks. How to switch, and back, is in the README next to the combined demo. (2026-10-17)

## Docs

- **`app/docs/ml-inventory.md` section 6, `app/APP.md` and `app/roadmap.md`** corrected to the leak-free numbers (`app/STATUS.md` is deleted upstream). (2026-10-13)
