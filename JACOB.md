# Jacob's changes

> What Jacob changed, newest first, so Jesse can catch up at a glance. One entry per change: what changed, and anything Jesse needs to do. Details are in the commits (`git log`).

## 2026-10-12

- **`analysis/DECISION_LOG.md` removed:** decisions now go in our change logs (`JACOB.md`, `JESSE.md`); references updated.
- **`analysis/JOB_GROWTH.md` removed:** the job growth steps and extension options are now tasks in `analysis/PRIORITY_QUEUE.md`.
- **`BRIEF.md` removed:** the hackathon brief is no longer in the repo or loaded into Claude sessions; the organizers' PDF stays local and git-ignored.
- **`DATA.md` removed:** we work from what we know about the data. The résumé-sourcing note the brief asks for is in README step 9; EDA_SDA_1 no longer writes schemas (section 5 and `write_schemas` are gone).
- **Hackathon brief moved to the top level:** `analysis/BRIEF.md` is now `BRIEF.md`, next to the README (CLAUDE.md imports it from there).
- **Change logs instead of Git rules:** `JACOB.md` and `JESSE.md` log each person's changes. No other rules on branches, pushes or merging.
- **Data reference moved to the top level:** `analysis/docs/data.md` is now `DATA.md`, next to the README. EDA_SDA_1 writes the schemas there.
- **BGE model download script:** `python -m fetch.get_bge` downloads `BAAI/bge-base-en-v1.5` (about 0.44 GB, no account needed) into `pretrained/bge-base-en-v1.5`, at the version the corpus's embeddings were made with. **Jesse:** run it once if you don't have the model.
- **`models/` renamed `pretrained/`:** it only holds downloaded models. `PRETRAINED_DIR` replaces `MODELS_DIR` in `fetch/paths.py`. **Jesse:** if you have a local `models/` folder, rename it to `pretrained/`.
- **`fetch/`: where every dataset comes from:** one download script per dataset (`get_predictleads`, `get_resumes`, `get_bge`), one standard folder each, a disk-space check, and nothing hard-coded (bucket names and profiles come from `.env`). The PredictLeads export can be downloaded to `data/raw/` with `python -m fetch.get_predictleads`; everything reads it locally when it's there and streams from S3 otherwise.
- **US only:** every listing with no country is dropped in EDA_SDA_1 8b.
- **Hackathon brief added** as `analysis/BRIEF.md` (the S3 location left out).
- **The merge plan is removed:** the two repositories are merged.
