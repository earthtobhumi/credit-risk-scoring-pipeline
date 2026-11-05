"""
Leakage-safe train / valid / holdout split.

  holdout (15%)  - never used for fitting, early stopping, or model selection.
                   Evaluated ONCE, on the final chosen model (Phase 3 part 2).
  valid   (~17%) - early stopping and model comparison.
  train   (~68%) - fitting.

All splits are stratified on TARGET (about 8% positives) so every part has the
same default rate. The assignment is written to SPLITS_FILE on first use and
re-read on every later run, so the holdout set stays the same even if a
different random seed or library version would have split differently. If the
set of applicants changes (new Gold data), the code refuses to reuse the old
split instead of silently mixing populations.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HOLDOUT_FRAC = 0.15
VALID_FRAC = 0.20  # of the non-holdout rows
SEED = 42


def assign_splits(
    ids: pd.Series,
    target: pd.Series,
    holdout_frac: float = HOLDOUT_FRAC,
    valid_frac: float = VALID_FRAC,
    seed: int = SEED,
) -> pd.DataFrame:
    df = pd.DataFrame({"SK_ID_CURR": ids.to_numpy(), "TARGET": target.to_numpy()})
    dev_idx, hold_idx = train_test_split(
        df.index, test_size=holdout_frac, stratify=df["TARGET"], random_state=seed
    )
    train_idx, valid_idx = train_test_split(
        dev_idx, test_size=valid_frac, stratify=df.loc[dev_idx, "TARGET"], random_state=seed
    )
    df["split"] = "train"
    df.loc[valid_idx, "split"] = "valid"
    df.loc[hold_idx, "split"] = "holdout"
    return df[["SK_ID_CURR", "split"]]


def load_or_create_splits(ids: pd.Series, target: pd.Series, path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        saved = pd.read_parquet(path)
        if set(saved["SK_ID_CURR"]) != set(ids):
            raise ValueError(
                f"{path} was made for a different set of applicants than the current "
                "Gold table. If the data really changed, delete that file to create a new "
                "split (this also resets what counts as the untouched holdout set)."
            )
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    splits = assign_splits(ids, target)
    splits.to_parquet(path, index=False)
    return splits
