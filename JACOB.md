# Jacob's changes

> What Jacob changed, by category, so Jesse can catch up at a glance. Categories and the lines in each are ranked most important first. Each line is the net result (a file moved and later deleted just reads "deleted"), with the date and anything Jesse needs to do. Details are in the commits (`git log`).

## Data and downloads

- **`models/` renamed `pretrained/`** (`PRETRAINED_DIR` in `fetch/paths.py`). **Jesse:** rename your local `models/` folder to `pretrained/`, or the app won't find the BGE model. (2026-10-12)
- **BGE model download:** `python -m fetch.get_bge` gets `BAAI/bge-base-en-v1.5` (about 0.44 GB, no account) at the version the corpus's embeddings were made with. **Jesse:** run it once if you don't have the model. (2026-10-12)
- **`fetch/`** is where every dataset comes from: one download script each (`get_predictleads`, `get_resumes`, `get_bge`), one standard folder each, a disk-space check, and nothing hard-coded (bucket names and profiles come from `.env`). The PredictLeads export can be downloaded to `data/raw/`; everything reads it locally when it's there and streams from S3 otherwise. (2026-10-12)

## Git workflow

- **No Git rules** beyond these change logs: work on your own branch, merge into `main` when ready, and log your changes here (README step 14). (2026-10-12)

## Docs

- **Deleted** the hackathon brief (`BRIEF.md`), the decision log (`analysis/DECISION_LOG.md`), the data reference (`analysis/docs/data.md`), the job growth plan (`analysis/JOB_GROWTH.md`) and the merge plan. (2026-10-12)
- **Decisions** are recorded in these change logs instead of a decision log. (2026-10-12)
- **Job growth** steps and extension options are now tasks in `analysis/PRIORITY_QUEUE.md`. (2026-10-12)
- **Candidate-profile sourcing** (the Kaggle résumés, as the brief asks) is a short note in README step 9. (2026-10-12)

## Analysis

- **US only:** EDA_SDA_1 8b drops every listing with no country. (2026-10-12)
- **EDA_SDA_1 no longer writes schemas** (section 5 and `write_schemas` in `src/pl_data.py` are gone). (2026-10-12)
