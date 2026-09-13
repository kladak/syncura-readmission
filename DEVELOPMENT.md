# Development notes

Quick personal scratchpad so I remember the local flow.

## Bootstrap once

```bash
./scripts/bootstrap.sh
```

If port 8000 is taken (happens a lot on shared boxes), use `--port 8001` and point the Vite proxy at it — or just hit the API with curl while iterating on SHAP.

## Why HGB is primary

Logistic sometimes edges ROC-AUC by a hair on this synthetic set, but TreeExplainer on HistGradientBoosting is much nicer for the dashboard. Both metrics stay in `models/metrics.json`.

## Leakage checklist I actually care about

- `patient_id` never enters `FEATURE_COLUMNS`
- generative `risk_logit` is dropped before CSV write
- target is only in eval / demo table audit column

## Not affiliated

Clean-room build. Do not vendor or copy from other Syncura-named repos.
