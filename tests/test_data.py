"""Leakage and schema tests for synthetic generator."""

from pathlib import Path

import pandas as pd

from src.syncura.data.generate import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_NAMES,
    assert_no_leakage,
    generate_cohort,
    stratified_split,
    write_dataset,
)


def test_feature_schema_excludes_forbidden():
    assert not (set(FEATURE_COLUMNS) & FORBIDDEN_FEATURE_NAMES)
    assert "patient_id" not in FEATURE_COLUMNS
    assert "readmitted_30d" not in FEATURE_COLUMNS
    assert "risk_logit" not in FEATURE_COLUMNS


def test_generate_and_leakage_report(tmp_path: Path):
    meta = write_dataset(tmp_path, n_patients=500, seed=7)
    assert meta["synthetic"] is True
    assert meta["leakage_report"]["leakage_ok"] is True
    cohort = pd.read_csv(tmp_path / "cohort.csv")
    assert "risk_logit" not in cohort.columns
    assert set(cohort["split"]) == {"train", "val", "test"}
    assert 0.05 < cohort["readmitted_30d"].mean() < 0.6


def test_assert_no_leakage_on_split_frame():
    df = stratified_split(generate_cohort(200, seed=1), seed=1)
    report = assert_no_leakage(df)
    assert report["n_rows"] == 200
