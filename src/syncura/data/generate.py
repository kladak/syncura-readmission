"""
Synthetic EHR-like tabular generator for educational 30-day readmission demos.

All rows are fabricated. No real hospital data. Fixed seed for reproducibility.
Leakage-safe: features are pre-discharge only; target is post-discharge label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "age",
    "sex_male",
    "insurance_medicare",
    "insurance_medicaid",
    "insurance_commercial",
    "length_of_stay",
    "discharge_home",
    "discharge_snf",
    "primary_dx_cardio",
    "primary_dx_resp",
    "primary_dx_renal",
    "primary_dx_other",
    "hemoglobin",
    "creatinine",
    "sodium",
    "glucose",
    "prior_admissions_365d",
    "ed_visits_180d",
    "med_count",
    "charlson_proxy",
    "heart_failure_flag",
    "diabetes_flag",
    "copd_flag",
]

FORBIDDEN_FEATURE_NAMES = {
    "readmitted_30d",
    "patient_id",
    "split",
    "risk_logit",
    "episode_id",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def generate_cohort(
    n_patients: int = 4000,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic discharge cohort with a sparse learnable risk function."""
    rng = np.random.default_rng(seed)

    age = rng.integers(22, 92, size=n_patients)
    sex_male = rng.binomial(1, 0.48, size=n_patients)

    # Insurance: multinomial -> one-hots
    ins = rng.choice(["medicare", "medicaid", "commercial", "other"], size=n_patients, p=[0.42, 0.18, 0.32, 0.08])
    insurance_medicare = (ins == "medicare").astype(int)
    insurance_medicaid = (ins == "medicaid").astype(int)
    insurance_commercial = (ins == "commercial").astype(int)

    length_of_stay = np.clip(rng.lognormal(mean=1.2, sigma=0.55, size=n_patients).round(), 1, 28).astype(int)

    disp = rng.choice(["home", "snf", "other"], size=n_patients, p=[0.72, 0.18, 0.10])
    discharge_home = (disp == "home").astype(int)
    discharge_snf = (disp == "snf").astype(int)

    dx = rng.choice(["cardio", "resp", "renal", "other"], size=n_patients, p=[0.28, 0.18, 0.12, 0.42])
    primary_dx_cardio = (dx == "cardio").astype(int)
    primary_dx_resp = (dx == "resp").astype(int)
    primary_dx_renal = (dx == "renal").astype(int)
    primary_dx_other = (dx == "other").astype(int)

    # Labs with mild age / comorbidity coupling later via flags
    hemoglobin = np.clip(rng.normal(12.8, 1.6, size=n_patients), 7.0, 17.5)
    creatinine = np.clip(rng.lognormal(mean=-0.05, sigma=0.35, size=n_patients), 0.4, 6.0)
    sodium = np.clip(rng.normal(138.5, 3.2, size=n_patients), 125, 150)
    glucose = np.clip(rng.lognormal(mean=4.7, sigma=0.28, size=n_patients), 70, 400)

    prior_admissions_365d = rng.poisson(0.55, size=n_patients).clip(0, 8)
    ed_visits_180d = rng.poisson(0.7, size=n_patients).clip(0, 10)
    med_count = np.clip(rng.poisson(6, size=n_patients) + (age > 70).astype(int), 0, 25)

    heart_failure_flag = rng.binomial(1, 0.12 + 0.004 * np.maximum(age - 55, 0) / 10, size=n_patients)
    diabetes_flag = rng.binomial(1, 0.22, size=n_patients)
    copd_flag = rng.binomial(1, 0.10 + 0.08 * primary_dx_resp, size=n_patients)
    charlson_proxy = (
        heart_failure_flag
        + diabetes_flag
        + copd_flag
        + (creatinine > 1.5).astype(int)
        + (age >= 75).astype(int)
        + prior_admissions_365d.clip(0, 3) // 2
    ).clip(0, 8)

    # Mild lab shifts with flags (still pre-discharge)
    hemoglobin = hemoglobin - 0.6 * heart_failure_flag - 0.3 * (age > 80)
    creatinine = creatinine + 0.35 * primary_dx_renal + 0.2 * diabetes_flag
    glucose = glucose + 25 * diabetes_flag

    # Sparse generative risk (educational signal, not clinical truth)
    logit = (
        -2.35
        + 0.018 * (age - 60)
        + 0.55 * (length_of_stay >= 7)
        + 0.70 * discharge_snf
        + 0.45 * primary_dx_cardio
        + 0.35 * primary_dx_renal
        + 0.40 * heart_failure_flag
        + 0.25 * diabetes_flag
        + 0.30 * copd_flag
        + 0.28 * charlson_proxy
        + 0.35 * prior_admissions_365d
        + 0.18 * ed_visits_180d
        + 0.04 * med_count
        + 0.55 * (creatinine > 1.8)
        + 0.40 * (hemoglobin < 10)
        + 0.25 * (sodium < 134)
        + 0.20 * insurance_medicaid
        - 0.15 * discharge_home
        + rng.normal(0, 0.55, size=n_patients)
    )
    risk = _sigmoid(logit)
    readmitted_30d = rng.binomial(1, risk)

    patient_id = [f"syn-{i:06d}" for i in range(n_patients)]

    df = pd.DataFrame(
        {
            "patient_id": patient_id,
            "age": age.astype(float),
            "sex_male": sex_male,
            "insurance_medicare": insurance_medicare,
            "insurance_medicaid": insurance_medicaid,
            "insurance_commercial": insurance_commercial,
            "length_of_stay": length_of_stay.astype(float),
            "discharge_home": discharge_home,
            "discharge_snf": discharge_snf,
            "primary_dx_cardio": primary_dx_cardio,
            "primary_dx_resp": primary_dx_resp,
            "primary_dx_renal": primary_dx_renal,
            "primary_dx_other": primary_dx_other,
            "hemoglobin": np.round(hemoglobin, 2),
            "creatinine": np.round(creatinine, 2),
            "sodium": np.round(sodium, 1),
            "glucose": np.round(glucose, 1),
            "prior_admissions_365d": prior_admissions_365d.astype(float),
            "ed_visits_180d": ed_visits_180d.astype(float),
            "med_count": med_count.astype(float),
            "charlson_proxy": charlson_proxy.astype(float),
            "heart_failure_flag": heart_failure_flag,
            "diabetes_flag": diabetes_flag,
            "copd_flag": copd_flag,
            "readmitted_30d": readmitted_30d,
            "risk_logit": np.round(logit, 4),  # audit only — stripped before training
        }
    )
    return df


def stratified_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    """Add split column with stratification on the target."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["split"] = ""

    for label in (0, 1):
        idx = out.index[out["readmitted_30d"] == label].to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        out.loc[train_idx, "split"] = "train"
        out.loc[val_idx, "split"] = "val"
        out.loc[test_idx, "split"] = "test"

    return out


def assert_no_leakage(df: pd.DataFrame) -> dict[str, Any]:
    """Runtime checks: forbidden columns not in feature set; split coverage; target present."""
    feature_present = [c for c in FEATURE_COLUMNS if c in df.columns]
    overlap = set(feature_present) & FORBIDDEN_FEATURE_NAMES
    if overlap:
        raise ValueError(f"Leakage: forbidden names in features: {overlap}")

    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected features: {missing}")

    if "readmitted_30d" not in df.columns:
        raise ValueError("Target readmitted_30d missing")

    if "patient_id" in FEATURE_COLUMNS:
        raise ValueError("patient_id must never be a model feature")

    splits = set(df["split"].unique()) if "split" in df.columns else set()
    report = {
        "n_rows": int(len(df)),
        "n_features": len(FEATURE_COLUMNS),
        "positive_rate": float(df["readmitted_30d"].mean()),
        "splits": {s: int((df["split"] == s).sum()) for s in sorted(splits)},
        "leakage_ok": True,
        "forbidden_in_features": [],
    }
    return report


def write_dataset(out_dir: Path, n_patients: int = 4000, seed: int = 42) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = generate_cohort(n_patients=n_patients, seed=seed)
    df = stratified_split(raw, seed=seed)
    report = assert_no_leakage(df)

    # Persist training table without generative logit (audit column)
    train_df = df.drop(columns=["risk_logit"])
    train_df.to_csv(out_dir / "cohort.csv", index=False)

    # Feature-only matrices per split for convenience
    for split in ("train", "val", "test"):
        part = train_df[train_df["split"] == split]
        part.to_csv(out_dir / f"{split}.csv", index=False)

    meta = {
        "synthetic": True,
        "disclaimer": "Synthetic EHR-like data for education only. Not real patients.",
        "n_patients": n_patients,
        "seed": seed,
        "feature_columns": FEATURE_COLUMNS,
        "target": "readmitted_30d",
        "leakage_report": report,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EHR-like readmission cohort")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--n-patients", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    meta = write_dataset(args.out, n_patients=args.n_patients, seed=args.seed)
    print(json.dumps(meta["leakage_report"], indent=2))
    print(f"Wrote synthetic cohort to {args.out}/")


if __name__ == "__main__":
    main()
