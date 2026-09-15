# syncura-readmission

Interpretable 30-day readmission risk on generated EHR-like tabular data: SHAP explanations, a FastAPI service and a small dashboard.

**Owner:** [Karim Ladak](https://github.com/kladak) (`kladak`)

![syncura-readmission dashboard: cohort list, 30-day risk gauge, SHAP factors](docs/syncura-dashboard.png)

Local Vite + FastAPI screenshot (2026-09-14): holdout patient `syn-000056`, HistGradientBoosting risk **47%** (above 0.35 threshold), TreeExplainer SHAP factors (`length_of_stay`, `discharge_snf`, …).

## Demo

1. Bootstrap data + model (see [Run Locally](#run-locally)).
2. Start the API, then the Vite dashboard.
3. Open `http://localhost:5173`.
4. Click a holdout patient in **Synthetic cohort**.
6. **Risk score** shows 30-day probability vs threshold; **Top contributing factors** are SHAP values from TreeExplainer on the trained HGB model.

The cohort table shows the stored label for inspection; the model receives only the feature columns.

## Results

After training, metrics are written to `models/metrics.json`. Example run (seed=42, n=4000):

| Metric (HGB primary, seed=42, n=4000) | Value |
|--------|-------|
| ROC-AUC | 0.666 |
| PR-AUC | 0.398 |
| Brier | 0.189 |
| Logistic ROC-AUC (baseline) | 0.671 |

Logistic regression reaches 0.671 test ROC-AUC and HGB 0.666 on this draw. HGB is primary because TreeExplainer drives the dashboard's explanation path; both models stay in `models/metrics.json`.

## Run Locally

### 1. Python env

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate data & train

```bash
python -m src.syncura.data.generate --out data --n-patients 4000 --seed 42
python -m src.syncura.model.train --data data --out models --seed 42
# or: ./scripts/bootstrap.sh
```

### 3. API

```bash
uvicorn src.syncura.api.main:app --reload --port 8000
```

- `GET /health`
- `GET /patients?limit=40` returns a holdout slice
- `GET /metrics`
- `POST /predict` takes `{ "features": { ... } }` or `{ "patient_id": "syn-000123" }`
- `POST /explain` takes the same body and returns the top SHAP factors

If port 8000 is taken:

```bash
uvicorn src.syncura.api.main:app --reload --port 8040
SYNCURA_API_URL=http://127.0.0.1:8040 npm run dev   # from frontend/
```

### 4. Dashboard

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL (default `http://localhost:5173`). The UI proxies `/api` → `http://localhost:8000` (override with `SYNCURA_API_URL`).

### Docker (optional)

```bash
docker compose up --build
```

API: `http://localhost:8000`, UI: `http://localhost:5173` (or the mapped port under compose).

## Data

`src/syncura/data/generate.py` creates patient records from configurable demographic,
utilization and lab feature distributions on a fixed seed. The generated dataset is used
for training and evaluation. [`PROVENANCE.md`](PROVENANCE.md) documents the generation
parameters and the leakage checks.

## What's inside

| Piece | Stack |
|-------|--------|
| Synthetic EHR generator | pandas / numpy |
| Model | scikit-learn HistGradientBoosting (+ logistic baseline) |
| Explanations | SHAP (TreeExplainer) |
| API | FastAPI |
| UI | Vite + React |
| Tests / CI | pytest + GitHub Actions |

## Project layout

```
SPEC.md
README.md
requirements.txt
src/syncura/
  data/generate.py
  model/train.py
  model/explain.py
  api/main.py
frontend/          # Vite React app
tests/
.github/workflows/ci.yml
docker-compose.yml
```

## License

MIT.
