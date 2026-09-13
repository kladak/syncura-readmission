"""SHAP explanation helpers for the primary tree or logistic model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


def explain_instance(
    model: Any,
    features: list[str],
    row: pd.DataFrame,
    background: pd.DataFrame | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """
    Return risk probability and top-k SHAP factors for a single-row DataFrame.
    Prefers TreeExplainer for HistGradientBoosting; KernelExplainer otherwise.
    """
    if list(row.columns) != features:
        row = row[features]

    proba = float(model.predict_proba(row)[0, 1])

    # Unwrap sklearn Pipeline logistic -> use Kernel on small bg
    estimator = model
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        # Kernel on full pipeline predictions
        if background is None or len(background) == 0:
            background = row
        bg = background[features].iloc[:40]
        explainer = shap.KernelExplainer(lambda X: model.predict_proba(X)[:, 1], bg.to_numpy())
        sv = np.array(explainer.shap_values(row.to_numpy(), nsamples=100))
        if sv.ndim > 1:
            sv = sv[0]
        base = float(explainer.expected_value)
    else:
        # HistGradientBoostingClassifier
        explainer = shap.TreeExplainer(model)
        sv_raw = explainer.shap_values(row)
        if isinstance(sv_raw, list):
            sv = np.array(sv_raw[1][0])  # positive class
        else:
            sv = np.array(sv_raw[0])
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.array(base).ravel()[-1])
        else:
            base = float(base)

    values = row.iloc[0].to_dict()
    pairs = [
        {
            "feature": features[i],
            "value": _jsonable(values[features[i]]),
            "contribution": float(sv[i]),
        }
        for i in range(len(features))
    ]
    pairs.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    top = pairs[:top_k]

    return {
        "risk_score": proba,
        "base_value": base,
        "top_factors": top,
        "all_factors": pairs,
    }


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v
