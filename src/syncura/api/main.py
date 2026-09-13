"""
FastAPI app: /health, /predict, /explain

Educational demo only — synthetic data, not a medical device.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.syncura.data.generate import FEATURE_COLUMNS
from src.syncura.model.explain import explain_instance

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
# Prefer CWD models/ when running from repo root
if (Path.cwd() / "models" / "primary.joblib").exists():
    MODELS_DIR = Path.cwd() / "models"

DISCLAIMER = (
    "Educational research demo using synthetic data only. "
    "Not a medical device. Not clinically validated. Not for care decisions."
)

app = FastAPI(
    title="syncura-readmission API",
    description=DISCLAIMER,
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: dict[str, float] | None = None
    patient_id: str | None = None
    threshold: float | None = Field(default=None, ge=0, le=1)


class Factor(BaseModel):
    feature: str
    value: float | int
    contribution: float


class PredictResponse(BaseModel):
    risk_score: float
    high_risk: bool
    threshold: float
    patient_id: str | None = None
    disclaimer: str = DISCLAIMER


class ExplainResponse(BaseModel):
    risk_score: float
    high_risk: bool
    threshold: float
    patient_id: str | None = None
    top_factors: list[Factor]
    disclaimer: str = DISCLAIMER


@lru_cache(maxsize=1)
def _load_artifacts() -> dict[str, Any]:
    primary_path = MODELS_DIR / "primary.joblib"
    if not primary_path.exists():
        raise FileNotFoundError(
            f"Missing {primary_path}. Run: python -m src.syncura.model.train"
        )
    model = joblib.load(primary_path)
    features = FEATURE_COLUMNS
    feat_path = MODELS_DIR / "feature_columns.json"
    if feat_path.exists():
        features = json.loads(feat_path.read_text())

    metrics: dict[str, Any] = {}
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    background = None
    bg_path = MODELS_DIR / "shap_background.csv"
    if bg_path.exists():
        background = pd.read_csv(bg_path)

    demo = None
    demo_path = MODELS_DIR / "demo_patients.csv"
    if demo_path.exists():
        demo = pd.read_csv(demo_path)

    return {
        "model": model,
        "features": features,
        "metrics": metrics,
        "background": background,
        "demo": demo,
        "threshold": float(metrics.get("threshold", 0.35)),
    }


def _row_from_request(req: PredictRequest, art: dict[str, Any]) -> tuple[pd.DataFrame, str | None]:
    features: list[str] = art["features"]
    if req.patient_id:
        demo: pd.DataFrame | None = art["demo"]
        if demo is None or demo.empty:
            raise HTTPException(404, "Demo cohort not loaded")
        hit = demo[demo["patient_id"] == req.patient_id]
        if hit.empty:
            raise HTTPException(404, f"Unknown patient_id: {req.patient_id}")
        return hit.iloc[[0]][features], req.patient_id

    if not req.features:
        raise HTTPException(400, "Provide features or patient_id")

    missing = [f for f in features if f not in req.features]
    if missing:
        raise HTTPException(400, f"Missing features: {missing[:8]}...")

    row = pd.DataFrame([{f: float(req.features[f]) for f in features}])
    return row, None


@app.get("/health")
def health() -> dict[str, Any]:
    ready = (MODELS_DIR / "primary.joblib").exists()
    return {
        "status": "ok" if ready else "degraded",
        "model_ready": ready,
        "disclaimer": DISCLAIMER,
        "synthetic": True,
    }


@app.get("/patients")
def list_patients(limit: int = 50) -> dict[str, Any]:
    art = _load_artifacts()
    demo: pd.DataFrame | None = art["demo"]
    if demo is None:
        return {"patients": [], "disclaimer": DISCLAIMER}
    cols = ["patient_id", "age", "length_of_stay", "charlson_proxy", "readmitted_30d"]
    cols = [c for c in cols if c in demo.columns]
    records = demo[cols].head(limit).to_dict(orient="records")
    return {"patients": records, "disclaimer": DISCLAIMER, "synthetic": True}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    art = _load_artifacts()
    m = dict(art["metrics"])
    m["disclaimer"] = DISCLAIMER
    return m


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    art = _load_artifacts()
    row, pid = _row_from_request(req, art)
    threshold = req.threshold if req.threshold is not None else art["threshold"]
    proba = float(art["model"].predict_proba(row)[0, 1])
    return PredictResponse(
        risk_score=round(proba, 4),
        high_risk=proba >= threshold,
        threshold=threshold,
        patient_id=pid,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: PredictRequest) -> ExplainResponse:
    art = _load_artifacts()
    row, pid = _row_from_request(req, art)
    threshold = req.threshold if req.threshold is not None else art["threshold"]
    result = explain_instance(
        art["model"],
        art["features"],
        row,
        background=art["background"],
        top_k=8,
    )
    return ExplainResponse(
        risk_score=round(result["risk_score"], 4),
        high_risk=result["risk_score"] >= threshold,
        threshold=threshold,
        patient_id=pid,
        top_factors=[Factor(**f) for f in result["top_factors"]],
    )
