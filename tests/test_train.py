"""Training smoke test on tiny synthetic cohort."""

from pathlib import Path

from src.syncura.data.generate import write_dataset
from src.syncura.model.train import train_models


def test_train_writes_metrics(tmp_path: Path):
    data = tmp_path / "data"
    models = tmp_path / "models"
    write_dataset(data, n_patients=600, seed=3)
    art = train_models(data, models, seed=3)
    assert (models / "primary.joblib").exists()
    assert (models / "metrics.json").exists()
    assert art["test"]["hist_gradient_boosting"]["roc_auc"] > 0.55
    assert art["synthetic"] is True
