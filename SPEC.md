# SPEC: Syncura Readmission

**Owner:** Karim Ladak (`kladak`)  
**Status:** v0  
**Data:** synthetic only. See [`PROVENANCE.md`](PROVENANCE.md).

## Purpose

Demonstrate an end-to-end pipeline for **interpretable 30-day hospital readmission risk** on **synthetic EHR-like tabular data**:

1. Generate synthetic patient/encounter features (demographics, labs, utilization).
2. Train a linear or gradient-boosted baseline.
3. Explain predictions with SHAP.
4. Serve risk + top factors via a small REST API.
5. Show results in a clinician-facing style dashboard (demo UI only).

## Non-goals / Hard constraints

- All rows come from the in-repo generator.
- Clinical validation and care-decision use are out of scope.

## Data (synthetic)

| Feature group | Examples |
|---------------|----------|
| Demographics | age, sex, insurance_type |
| Encounter | length_of_stay, discharge_disposition, primary_dx_group |
| Labs (last encounter) | hemoglobin, creatinine, sodium, glucose |
| Utilization | prior_admissions_365d, ed_visits_180d, med_count |
| Comorbidity proxies | charlson_proxy, heart_failure_flag, diabetes_flag, copd_flag |

**Target:** `readmitted_30d` ∈ {0, 1}, drawn from a known sparse risk function plus noise so the model has a learnable signal.

**Splits:** stratified train / val / test (70 / 15 / 15). Leakage checks assert that `FEATURE_COLUMNS` excludes post-discharge labels and `patient_id`.

## Model

- Primary: HistGradientBoostingClassifier (sklearn), a strong tabular baseline that TreeExplainer supports directly.
- Optional baseline: LogisticRegression with standardized numerics for calibration reference.
- Metrics reported on **synthetic holdout only**: ROC-AUC, PR-AUC, Brier, calibration sketch.
- SHAP: TreeExplainer for tree model; KernelExplainer fallback on a small background set if needed.

## API (FastAPI)

| Endpoint | Method | Behavior |
|----------|--------|----------|
| `/health` | GET | liveness |
| `/predict` | POST | risk score + class threshold flag |
| `/explain` | POST | risk + top-k SHAP factors (feature, value, contribution) |

Request body: feature dict matching the training schema (or `patient_id` lookup against a bundled synthetic cohort for the dashboard).

## Dashboard

Simple Vite + React UI:

- Patient table (synthetic cohort sample)
- Selected patient risk gauge / score
- Ranked contributing factors from `/explain`
- A banner noting the cohort is generated

## Reproducibility

- Fixed random seeds for data + training.
- `requirements.txt` / lock-friendly pins where practical.
- Scripts: `generate_data.py`, `train.py`, `serve` via uvicorn.
- Tests: data leakage guards, API contract smoke tests, schema checks.
- CI: GitHub Actions running pytest.


