# syncura-readmission

**Clean-room educational demo** of interpretable 30-day readmission risk on **synthetic** EHR-like tabular data — SHAP explanations, FastAPI, and a small clinician-facing dashboard.

> **Not a medical device.** Synthetic data only. Not clinically validated. Do not use for care decisions.

**Owner:** [Karim Ladak](https://github.com/kladak) (`kladak`)

## Clean-room / affiliation notice

This repository is an **independent implementation** owned solely by Karim Ladak. It is **not affiliated with**, derived from, or a fork of:

- Fizan-Feroz/SynCura  
- Anya198/SyncuraV0  
- SpeciaList, Precision Cardiology, or any other Syncura-named codebase  

Inspiration is limited to a public resume-style description of the *problem shape* (readmission risk + interpretability + API + dashboard). **No code was copied or cloned** from those projects.

See [SPEC.md](./SPEC.md) for scope and constraints.

## What's inside

| Piece | Stack |
|-------|--------|
| Synthetic EHR generator | pandas / numpy |
| Model | scikit-learn HistGradientBoosting (+ logistic baseline) |
| Explanations | SHAP (TreeExplainer) |
| API | FastAPI |
| UI | Vite + React |
| Tests / CI | pytest + GitHub Actions |

## Quick start

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
```

### 3. API

```bash
uvicorn src.syncura.api.main:app --reload --port 8000
```

- `GET /health`
- `POST /predict` — `{ "features": { ... } }` or `{ "patient_id": "syn-000123" }`
- `POST /explain` — same body; returns top SHAP factors

### 4. Dashboard

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL (default `http://localhost:5173`). The UI proxies `/api` → `http://localhost:8000`.

### Docker (optional)

```bash
docker compose up --build
```

API: `http://localhost:8000` · UI: `http://localhost:5173` (or mapped port in compose).

## Synthetic metrics (holdout)

After training, metrics are written to `models/metrics.json`. Example run (seed=42, n=4000):

| Metric | Value |
|--------|-------|
| ROC-AUC | *(filled after train)* |
| PR-AUC | *(filled after train)* |
| Brier | *(filled after train)* |

These numbers describe **synthetic** signal only — they are not clinical performance.

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

MIT — educational use. No warranty. Not for clinical deployment.

