from __future__ import annotations

"""Per-call explainability for the dialogue logistic regression.

The production model is `Pipeline([StandardScaler, CalibratedClassifierCV(LogisticRegression)])`.
Since it's linear, "why did it decide this" is just weight * standardized_value
per feature, summed to the logit. This mirrors the aggregate weight table in
docs/FEATURES.md, but computed for one real call instead of the whole dataset.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from detector.features import vectorize


def _mean_coef_and_intercept(model) -> tuple[np.ndarray, float] | None:
    clf = model
    if isinstance(model, Pipeline):
        clf = model.named_steps.get("clf", model)
    if not isinstance(clf, CalibratedClassifierCV):
        inner = clf
        if hasattr(inner, "coef_"):
            return inner.coef_.ravel(), float(inner.intercept_.ravel()[0])
        return None
    coefs, intercepts = [], []
    for est in getattr(clf, "calibrated_classifiers_", []):
        inner = getattr(est, "estimator", None) or getattr(est, "base_estimator", None)
        if inner is not None and hasattr(inner, "coef_"):
            coefs.append(inner.coef_.ravel())
            intercepts.append(float(inner.intercept_.ravel()[0]))
    if not coefs:
        return None
    return np.mean(coefs, axis=0), float(np.mean(intercepts))


def explain(feat_dict: dict[str, float], bundle: dict) -> list[dict] | None:
    """Per-feature contribution (weight * standardized value) for one call.

    Returns rows sorted by |contribution| descending, or None if the bundle's
    model isn't a linear pipeline we know how to decompose (e.g. HGB).
    """
    model = bundle["model"]
    names = bundle["feature_names"]
    coef_intercept = _mean_coef_and_intercept(model)
    if coef_intercept is None:
        return None
    coef, intercept = coef_intercept

    scaler = model.named_steps.get("scaler") if isinstance(model, Pipeline) else None
    x = vectorize(feat_dict, names).reshape(1, -1)
    x_scaled = scaler.transform(x)[0] if scaler is not None else x[0]

    contributions = coef * x_scaled
    rows = [
        {
            "feature": name,
            "value": round(float(raw), 4),
            "standardized_value": round(float(std), 4),
            "weight": round(float(w), 4),
            "contribution": round(float(c), 4),
        }
        for name, raw, std, w, c in zip(names, x[0], x_scaled, coef, contributions)
    ]
    rows.sort(key=lambda r: -abs(r["contribution"]))
    return rows
