import numpy as np
import pandas as pd
import lightgbm as lgb

from config import METEO_COLS, AV_CLIP_LO, AV_CLIP_HI, LGBM_JOBS
from logging_setup import get_logger

log = get_logger("adversarial")


def compute_region_adversarial_weights(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, float]:
    """Compute per-region adversarial weights to reweight train toward the test distribution.

    Steps:
      1. Compute per-region meteorological summary statistics (mean, std, q25, q75).
      2. Train a binary LightGBM: train-region stats → label 0, test → label 1.
      3. Predict P(test-like) for each train region.
      4. weight = P / (1 - P), clipped and normalized to mean=1.

    Regions whose weather distribution is closer to test get higher weights.
    """
    log.info("Computing adversarial region weights ...")

    available_cols = [c for c in METEO_COLS if c in train.columns and c in test.columns]
    if not available_cols:
        log.warning("  No usable METEO_COLS; returning uniform weights of 1.0")
        return {str(r): 1.0 for r in train["region_id"].unique()}

    # ── Per-region meteorological statistics ──────────────────────────────────
    def _region_stats(df: pd.DataFrame) -> pd.DataFrame:
        grp = df.groupby("region_id")[available_cols]
        mean_df = grp.mean().add_suffix("_mean")
        std_df  = grp.std().add_suffix("_std")
        q25_df  = grp.quantile(0.25).add_suffix("_q25")
        q75_df  = grp.quantile(0.75).add_suffix("_q75")
        return pd.concat([mean_df, std_df, q25_df, q75_df], axis=1).fillna(0)

    train_stats = _region_stats(train)
    test_stats  = _region_stats(test)

    train_stats["label"] = 0
    test_stats["label"]  = 1

    feat_cols = [c for c in train_stats.columns if c != "label"]
    combined  = pd.concat([train_stats, test_stats]).fillna(0)

    X_all = combined[feat_cols].values.astype(np.float32)
    y_all = combined["label"].values.astype(np.int8)

    n_train_reg = len(train_stats)
    n_test_reg  = len(test_stats)
    log.info(f"  Train regions={n_train_reg}, Test regions={n_test_reg}, "
             f"features per region={len(feat_cols)}")

    # ── Binary LGBM classifier ─────────────────────────────────────────────────
    clf = lgb.LGBMClassifier(
        n_estimators      = 300,
        num_leaves        = 31,
        learning_rate     = 0.05,
        min_child_samples = 5,
        feature_fraction  = 0.8,
        bagging_fraction  = 0.8,
        bagging_freq      = 5,
        verbose           = -1,
        n_jobs            = LGBM_JOBS,
        random_state      = 42,
    )
    clf.fit(X_all, y_all)

    # ── Predict P(test-like) for each train region ────────────────────────────
    X_train_reg = train_stats[feat_cols].values.astype(np.float32)
    proba = clf.predict_proba(X_train_reg)[:, 1]

    train_proba_mean = float(proba.mean())
    log.info(f"  Train regions P(test-like) mean={train_proba_mean:.3f}  "
             f"(0.5 = no shift, >0.5 = train shifted toward test)")

    # ── Compute weight = P/(1-P), clip, and normalize ─────────────────────────
    eps = 1e-6
    raw_weights  = np.clip(proba / (1.0 - proba + eps), AV_CLIP_LO, AV_CLIP_HI)
    norm_weights = raw_weights / raw_weights.mean()

    region_ids  = train_stats.index.tolist()
    weight_dict = {str(r): float(w) for r, w in zip(region_ids, norm_weights)}

    w_arr = np.array(list(weight_dict.values()))
    log.info(f"  Adversarial weights: mean={w_arr.mean():.3f}  "
             f"std={w_arr.std():.3f}  "
             f"[min={w_arr.min():.3f}, max={w_arr.max():.3f}]")

    return weight_dict


def apply_adversarial_weights(
    sample_weights: np.ndarray,
    groups: np.ndarray,
    weight_dict: dict[str, float],
) -> np.ndarray:
    """Multiply existing sample weights by per-region adversarial weights and renormalize."""
    av_w = np.array([weight_dict.get(str(g), 1.0) for g in groups], dtype=np.float32)
    combined = sample_weights * av_w
    combined = combined / combined.mean()
    return combined.astype(np.float32)
