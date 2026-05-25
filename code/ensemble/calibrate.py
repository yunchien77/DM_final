"""Isotonic calibration for the ensemble blend.

Fit one IsotonicRegression per horizon on val (blend_pred -> truth), then
apply to test-side blend. Clipped to [0, 5]. Cheap mean-correction that often
shaves a few thousandths off MAE.

Usage:
    from ensemble.calibrate import fit_isotonic_per_horizon, apply_isotonic

    iso_models = fit_isotonic_per_horizon(blend_val, truth_val)  # blend_val: (N, H)
    test_calibrated = apply_isotonic(iso_models, blend_test)
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic_per_horizon(
    blend_val: np.ndarray,  # (N, H)
    truth_val: np.ndarray,  # (N, H)
    y_min: float = 0.0,
    y_max: float = 5.0,
) -> list[IsotonicRegression]:
    """One monotone non-decreasing isotonic per horizon. The output is clipped
    to [y_min, y_max] by `out_of_bounds='clip'` + value clip below."""
    n, h = blend_val.shape
    assert truth_val.shape == blend_val.shape
    models = []
    for j in range(h):
        ir = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds="clip")
        ir.fit(blend_val[:, j], truth_val[:, j])
        models.append(ir)
    return models


def apply_isotonic(
    models: list[IsotonicRegression],
    preds: np.ndarray,  # (N, H)
    y_min: float = 0.0,
    y_max: float = 5.0,
) -> np.ndarray:
    out = np.zeros_like(preds, dtype=np.float32)
    for j, ir in enumerate(models):
        out[:, j] = ir.transform(preds[:, j])
    return np.clip(out, y_min, y_max)


def macro_mae(preds: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([np.mean(np.abs(preds[:, h] - truth[:, h]))
                          for h in range(preds.shape[1])]))


def calibrate_blend(
    blend_val: np.ndarray, truth_val: np.ndarray,
    blend_test: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Convenience wrapper: fit isotonic on val, transform test, report deltas."""
    pre = macro_mae(blend_val, truth_val)
    models = fit_isotonic_per_horizon(blend_val, truth_val)
    cal_val = apply_isotonic(models, blend_val)
    post = macro_mae(cal_val, truth_val)
    cal_test = apply_isotonic(models, blend_test)
    return cal_test, {"val_macro_pre": pre, "val_macro_post": post, "delta": pre - post}
