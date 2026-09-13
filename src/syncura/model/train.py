"""Train logistic + HistGradientBoosting baselines on synthetic cohort; save artifacts + metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.syncura.data.generate import FEATURE_COLUMNS, FORBIDDEN_FEATURE_NAMES


def load_split(data_dir: Path, split: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    path = data_dir / f"{split}.csv"
    if not path.exists():
        # fall back to cohort.csv
        cohort = pd.read_csv(data_dir / "cohort.csv")
        df = cohort[cohort["split"] == split].copy()
    else:
        df = pd.read_csv(path)

    leak = set(FEATURE_COLUMNS) & FORBIDDEN_FEATURE_NAMES
    if leak:
        raise RuntimeError(f"Feature schema leakage: {leak}")
    if "patient_id" in FEATURE_COLUMNS:
        raise RuntimeError("patient_id must not be a feature")

    X = df[FEATURE_COLUMNS]
    y = df["readmitted_30d"].to_numpy().astype(int)
    return X, y, FEATURE_COLUMNS


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }
    try:
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=8, strategy="quantile")
        metrics["calibration"] = {
            "fraction_positives": [float(x) for x in frac_pos],
            "mean_predicted": [float(x) for x in mean_pred],
        }
    except ValueError:
        metrics["calibration"] = None
    return metrics


def train_models(data_dir: Path, out_dir: Path, seed: int = 42) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train, features = load_split(data_dir, "train")
    X_val, y_val, _ = load_split(data_dir, "val")
    X_test, y_test, _ = load_split(data_dir, "test")

    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    logistic.fit(X_train, y_train)

    hgb = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=200,
        min_samples_leaf=25,
        l2_regularization=0.1,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
    )
    hgb.fit(X_train, y_train)

    # Select primary model by val ROC-AUC
    val_scores = {
        "logistic": evaluate(y_val, logistic.predict_proba(X_val)[:, 1]),
        "hist_gradient_boosting": evaluate(y_val, hgb.predict_proba(X_val)[:, 1]),
    }
    primary = max(val_scores, key=lambda k: val_scores[k]["roc_auc"])
    primary_model = logistic if primary == "logistic" else hgb

    test_metrics = {
        "logistic": evaluate(y_test, logistic.predict_proba(X_test)[:, 1]),
        "hist_gradient_boosting": evaluate(y_test, hgb.predict_proba(X_test)[:, 1]),
    }

    joblib.dump(logistic, out_dir / "logistic.joblib")
    joblib.dump(hgb, out_dir / "hgb.joblib")
    joblib.dump(primary_model, out_dir / "primary.joblib")

    # Small background sample for SHAP / API
    bg = X_train.sample(n=min(100, len(X_train)), random_state=seed)
    bg.to_csv(out_dir / "shap_background.csv", index=False)

    # Bundle a dashboard cohort slice (test set with ids + labels for demo)
    cohort = pd.read_csv(data_dir / "cohort.csv")
    demo = cohort[cohort["split"] == "test"].head(200).drop(columns=["risk_logit"], errors="ignore")
    demo.to_csv(out_dir / "demo_patients.csv", index=False)

    artifact = {
        "synthetic": True,
        "disclaimer": "Metrics on synthetic holdout only. Not clinical validation.",
        "seed": seed,
        "features": features,
        "primary_model": primary,
        "val": val_scores,
        "test": test_metrics,
        "threshold": 0.35,
    }
    (out_dir / "metrics.json").write_text(json.dumps(artifact, indent=2))
    (out_dir / "feature_columns.json").write_text(json.dumps(features, indent=2))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    art = train_models(args.data, args.out, seed=args.seed)
    print(json.dumps({"primary": art["primary_model"], "test": art["test"]}, indent=2))


if __name__ == "__main__":
    main()
