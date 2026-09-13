# SPEC — Syncura Readmission (Clean-Room Educational Demo)

**Owner:** Karim Ladak (`kladak`)  
**Status:** Educational research prototype  
**Affiliation:** Independent clean-room build. **Not affiliated with** Fizan-Feroz/SynCura, Anya198/SyncuraV0, SpeciaList, Precision Cardiology, or any other Syncura-named project.

## Purpose

Demonstrate an end-to-end pipeline for **interpretable 30-day hospital readmission risk** on **synthetic EHR-like tabular data**:

1. Generate synthetic patient/encounter features (demographics, labs, utilization).
2. Train a transparent baseline (logistic regression or gradient boosting).
3. Explain predictions with SHAP.
4. Serve risk + top factors via a small REST API.
5. Show results in a clinician-facing style dashboard (demo UI only).

## Non-goals / Hard constraints

- **No real PHI / hospital data.** All rows are synthetic and generated in-repo.
- **Not a medical device.** No clinical validation claims. Not for care decisions.
- **No code reuse** from prior Syncura-named repositories. Inspiration is limited to a public resume-style description of the *problem shape* (readmission risk + SHAP + API + dashboard).
- No scraping of clinical guidelines or proprietary datasets.

## Data (synthetic)

| Feature group | Examples |
|---------------|----------|
| Demographics | age, sex, insurance_type |
| Encounter | length_of_stay, discharge_disposition, primary_dx_group |
| Labs (last encounter) | hemoglobin, creatinine, sodium, glucose |
| Utilization | prior_admissions_365d, ed_visits_180d, med_count |
| Comorbidity proxies | charlson_proxy, heart_failure_flag, diabetes_flag, copd_flag |

**Target:** `readmitted_30d` ∈ {0, 1}, generated from a known sparse risk function + noise so the model has a learnable signal without claiming realism.

**Splits:** stratified train / val / test (70 / 15 / 15). Leakage checks: no post-discharge labels in features; patient_id not used as a feature; time-ordered within synthetic episode if applicable.

## Model

- Primary: HistGradientBoostingClassifier (sklearn) — strong tabular baseline, TreeExplainer-friendly via shap.
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
- Prominent educational disclaimer banner

## Reproducibility

- Fixed random seeds for data + training.
- `requirements.txt` / lock-friendly pins where practical.
- Scripts: `generate_data.py`, `train.py`, `serve` via uvicorn.
- Tests: data leakage guards, API contract smoke tests, schema checks.
- CI: GitHub Actions — lint-light + pytest.

## Disclaimer (always surface)

> This project is an **educational research demo** using **synthetic data only**. It is **not** a medical device, **not** clinically validated, and **must not** be used for diagnosis, treatment, or discharge planning.

