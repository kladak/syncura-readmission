# Data provenance

`src/syncura/data/generate.py` creates every record in this repository from a fixed seed.

`data/meta.json` records the generation parameters alongside a leakage report: row and
feature counts, split sizes, the target's positive rate, and `forbidden_in_features`, which
must stay empty. The generative `risk_logit` and `patient_id` are dropped before the CSV is
written, so `FEATURE_COLUMNS` carries only the 23 modelling features. The target is used in
evaluation and shown in the dashboard's audit column.
