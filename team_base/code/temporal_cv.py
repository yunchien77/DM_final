import numpy as np
from typing import Iterator
from logging_setup import get_logger

log = get_logger("temporal_cv")


class TemporalKFold:
    """Walk-forward temporal cross-validator.

    Splits samples by their window end-time rather than by region, so each
    validation set is always strictly in the future relative to its training set.
    A purge gap (gap_days) removes samples whose windows overlap the boundary,
    preventing leakage from sliding-window features.

    Parameters
    ----------
    n_splits : int
        Number of time blocks.  Produces n_splits-1 walk-forward folds.
    gap_days : int
        Minimum calendar-day gap between the last training sample and the first
        validation sample.  Uses ordinal-day time_keys, so the gap is exact
        regardless of sample density.
    """

    def __init__(self, n_splits: int = 5, gap_days: int = 91):
        self.n_splits = n_splits
        self.gap_days = gap_days

    def split(
        self,
        time_keys: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, val_idx) pairs sorted by time.

        Parameters
        ----------
        time_keys : np.ndarray
            Ordinal date (integer days) for each sample.
        """
        n = len(time_keys)
        sorted_pos    = np.argsort(time_keys, kind="stable")
        sorted_times  = time_keys[sorted_pos]
        block_size    = n // self.n_splits

        log.info(
            f"TemporalKFold: n={n:,}, splits={self.n_splits}, "
            f"block_size≈{block_size:,}, gap_days={self.gap_days}"
        )

        for fold in range(1, self.n_splits):
            val_start_pos = fold * block_size
            val_end_pos   = (fold + 1) * block_size if fold < self.n_splits - 1 else n

            val_start_time = sorted_times[val_start_pos]
            cutoff_time    = val_start_time - self.gap_days

            # Binary search: last position with time ≤ cutoff_time
            train_end_pos = int(np.searchsorted(sorted_times, cutoff_time, side="right"))

            if train_end_pos <= 0:
                log.warning(f"  Fold {fold}: train_end_pos={train_end_pos} <= 0, skip")
                continue

            train_sorted_idx = sorted_pos[:train_end_pos]
            val_sorted_idx   = sorted_pos[val_start_pos:val_end_pos]

            actual_gap_days = int(val_start_time - sorted_times[train_end_pos - 1])

            log.info(
                f"  Fold {fold}/{self.n_splits-1}: "
                f"train={len(train_sorted_idx):,}  val={len(val_sorted_idx):,}  "
                f"purge_gap={actual_gap_days}d (target={self.gap_days}d)"
            )
            yield train_sorted_idx, val_sorted_idx
