"""API contract smoke tests (requires trained artifacts in tmp)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.syncura.data.generate import FEATURE_COLUMNS, write_dataset
from src.syncura.model.train import train_models


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("syncura")
    data = root / "data"
    models = root / "models"
    write_dataset(data, n_patients=800, seed=42)
    train_models(data, models, seed=42)
    return models


@pytest.fixture
def client(trained, monkeypatch):
    import src.syncura.api.main as api_main

    monkeypatch.setattr(api_main, "MODELS_DIR", trained)
    api_main._load_artifacts.cache_clear()
    return TestClient(api_main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["synthetic"] is True


def test_predict_and_explain_with_features(client):
    features = {f: 0.0 for f in FEATURE_COLUMNS}
    features.update(
        {
            "age": 78,
            "length_of_stay": 9,
            "discharge_snf": 1,
            "heart_failure_flag": 1,
            "creatinine": 2.1,
            "hemoglobin": 9.5,
            "prior_admissions_365d": 3,
            "charlson_proxy": 4,
        }
    )
    r = client.post("/predict", json={"features": features})
    assert r.status_code == 200
    pred = r.json()
    assert 0 <= pred["risk_score"] <= 1

    r2 = client.post("/explain", json={"features": features})
    assert r2.status_code == 200
    exp = r2.json()
    assert len(exp["top_factors"]) >= 1
    assert "feature" in exp["top_factors"][0]


def test_patients_list(client):
    r = client.get("/patients?limit=10")
    assert r.status_code == 200
    assert "patients" in r.json()
