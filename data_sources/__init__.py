"""Where the project's data comes from and where it lives, shared by analysis/ and app/.

Every dataset reaches the code through here, never through a path or setting written into a notebook or script:
    data_sources.paths         REPO_ROOT, DATA_DIR and the standard folders (one definition for both halves)
    data_sources.predictleads  the PredictLeads export: S3 settings from .env, local copies in data/raw/
    data_sources.downloads     the disk-space check and question every download runs first
    data_sources.get_*         one script per dataset (python -m data_sources.get_<name>, from the repository root)

Adding a dataset: the checklist in CLAUDE.md ("Adding a new dataset").
"""
