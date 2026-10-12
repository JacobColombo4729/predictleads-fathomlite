"""Where the project's data comes from and where it lives, shared by analysis/ and app/.

Every dataset reaches the code through here, never through a path or setting written into a notebook or script:
    fetch.paths         REPO_ROOT, DATA_DIR and the standard folders (one definition for both halves)
    fetch.predictleads  the PredictLeads export: S3 settings from .env, local copies in data/raw/
    fetch.downloads     the disk-space check and question every download runs first
    fetch.get_*         one script per dataset (python -m fetch.get_<name>, from the repository root)

Adding a dataset: the checklist in CLAUDE.md ("Adding a new dataset").
"""
