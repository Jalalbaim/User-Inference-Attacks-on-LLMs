# data/

Dataset preparation, one sub-folder per dataset (`enron/` first; `reddit/`, CC News and ArXiv later).
**Status:** empty. How the data is collected and prepared is still to be decided with the user.

Rules that already hold (from `docs/experiment_plan_enron_gptneo125m.md`):
- One "user" per dataset is fixed up front (Enron: the sender address, as in Kandpal et al.).
- Splits are by **user**, never by document within a user. Splits are seeded and saved.
- Each prepared dataset reports its own counts (users, documents per user) after every filter.
- Raw downloads go to `data/raw/` and prepared files to `data/processed/`. Both are git-ignored and
  are never packed into Kaggle builds. On Kaggle, prepared data lives in `$UIA_DATA_DIR`.
