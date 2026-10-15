# Jesse's changes

> What Jesse changed, by category, so Jacob can catch up at a glance. Categories and the lines in each are ranked most important first. Each line is the net result (a file moved and later deleted just reads "deleted"), with the date and anything Jacob needs to do. Details are in the commits (`git log`).

## Machine learning

- **Role-family classifier fixed: its 94.7% was label leakage** (branch `ml/fix-classifier-label-leak`, not merged). The model read the O*NET occupation name inside the saved BGE text, and the labels come from that name. Leak-free accuracy is about 72% (TF-IDF on titles; BGE title embeddings 70.5%; majority class 30%), over 3 company-held-out splits. `app/code/role_family_classifier.py` now embeds titles only (cached in `data/matching/role_family_classifier_cache/`), reports mean over splits, has an optional `--leak-control`, and no longer predicts families for resumes (unvalidated). New guard tests: `app/tests/test_role_family_classifier.py`. **Jacob:** do not quote 94.7%. The Compass still imports `FAMILY_OF_OCCUPATION` unchanged. (2026-10-13)

## Docs

- **`app/docs/ml-inventory.md` section 6, `app/STATUS.md` and `app/roadmap.md`** corrected to the leak-free numbers. (2026-10-13)
