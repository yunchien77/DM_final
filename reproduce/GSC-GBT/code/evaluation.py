import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from logging_setup import get_logger

log = get_logger("evaluation")


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label: str = "") -> dict:
    """Compute MAE, RMSE, R², MASE, bias, P95 error, and per-class MAE."""
    y_true = np.clip(np.asarray(y_true, dtype=np.float32), 0, 5)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float32), 0, 5)

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    p95  = float(np.percentile(np.abs(y_true - y_pred), 95))

    # MASE relative to a naive (previous score) baseline
    naive_mae = float(np.mean(np.abs(np.diff(y_true)))) if len(y_true) > 1 else 1.0
    mase = mae / max(naive_mae, 1e-6)

    mae_by_score = {}
    for sc in range(6):
        mask = (y_true == sc)
        if mask.sum() > 0:
            mae_by_score[int(sc)] = float(mean_absolute_error(
                y_true[mask], y_pred[mask]))
        else:
            mae_by_score[int(sc)] = None

    metrics = dict(
        label=label,
        n=len(y_true),
        mae=mae,
        rmse=rmse,
        r2=r2,
        bias=bias,
        p95_abs_error=p95,
        mase=mase,
        mae_by_score=mae_by_score,
    )
    return metrics


def print_metrics(m: dict):
    log.info(f"── Metrics [{m['label']}] n={m['n']:,} ──")
    log.info(f"  MAE  = {m['mae']:.4f}   RMSE = {m['rmse']:.4f}")
    log.info(f"  R²   = {m['r2']:.4f}   Bias = {m['bias']:+.4f}")
    log.info(f"  MASE = {m['mase']:.4f}  P95_err = {m['p95_abs_error']:.4f}")
    log.info("  MAE by score:")
    for sc, v in m["mae_by_score"].items():
        s = f"{v:.4f}" if v is not None else "N/A"
        log.info(f"    score={sc}: {s}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-region MAE
# ─────────────────────────────────────────────────────────────────────────────

def per_region_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    """Return per-region MAE sorted descending (highest-error regions first)."""
    y_true = np.clip(y_true, 0, 5)
    y_pred = np.clip(y_pred, 0, 5)
    rows = []
    for region in np.unique(groups):
        mask = groups == region
        if mask.sum() == 0:
            continue
        rows.append(dict(
            region_id=region,
            n=int(mask.sum()),
            mae=float(mean_absolute_error(y_true[mask], y_pred[mask])),
            mean_true=float(np.mean(y_true[mask])),
            mean_pred=float(np.mean(y_pred[mask])),
        ))
    df = pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fold summary
# ─────────────────────────────────────────────────────────────────────────────

def fold_summary(fold_maes: list[float]) -> dict:
    arr = np.array(fold_maes)
    return dict(
        fold_maes=fold_maes,
        mean=float(arr.mean()),
        std=float(arr.std()),
        min=float(arr.min()),
        max=float(arr.max()),
        cv_stability=float(arr.std() / arr.mean()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────

def save_eval_report(report: dict, path: Path):
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_convert(report), f, indent=2, ensure_ascii=False)
    log.info(f"eval report saved → {path}")
