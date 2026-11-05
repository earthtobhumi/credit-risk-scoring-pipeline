"""
Training data access: Feast offline store -> model-ready matrices.

Training must read features through Feast's `get_historical_features`, the same
feature definitions the serving endpoint will use, so training and serving can
never silently disagree about how a feature is computed. The entity dataframe
carries SK_ID_CURR, TARGET and an event timestamp (see src/features/feast_demo.py).

The pulled frame is cached on disk and rebuilt automatically when the Feast
Parquet export is newer than the cache.
"""
from __future__ import annotations

import pandas as pd

from src.common import config

NON_FEATURE_COLUMNS = ["SK_ID_CURR", "TARGET", "event_timestamp"]


def load_training_frame(refresh: bool = False) -> pd.DataFrame:
    cache, source = config.TRAINING_FRAME_CACHE, config.FEAST_FEATURES_PARQUET
    if cache.exists() and not refresh and cache.stat().st_mtime >= source.stat().st_mtime:
        return pd.read_parquet(cache)

    from feast import FeatureStore  # imported lazily: slow, and not needed for cached runs

    store = FeatureStore(repo_path=str(config.FEAST_REPO_DIR))
    entity_df = pd.read_parquet(source, columns=["SK_ID_CURR", "TARGET"])
    entity_df["event_timestamp"] = pd.Timestamp.now(tz="UTC")
    frame = store.get_historical_features(
        entity_df=entity_df, features=store.get_feature_service(config.FEAST_FEATURE_SERVICE)
    ).to_df()

    _check_frame(frame, expected_rows=len(entity_df))
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def _check_frame(frame: pd.DataFrame, expected_rows: int) -> None:
    """Fail loudly on the two silent failure modes of a point-in-time join."""
    if len(frame) != expected_rows or not frame["SK_ID_CURR"].is_unique:
        raise ValueError(
            f"Feast returned {len(frame)} rows for {expected_rows} applicants "
            f"(unique ids: {frame['SK_ID_CURR'].is_unique}). A point-in-time join must "
            "return exactly one row per applicant."
        )
    # age_years is never null by construction, so nulls there mean features expired
    # or the entity timestamps are earlier than the feature timestamps.
    if frame["age_years"].isna().mean() > 0.01:
        raise ValueError(
            "Feast returned NULL features for more than 1% of applicants. Check the "
            "feature view TTL and that event_timestamp is later than the feature run time."
        )


def feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a Feast frame into (X, y). Booleans become 0/1; NaNs stay NaN (both
    XGBoost and LightGBM route missing values natively)."""
    X = frame.drop(columns=NON_FEATURE_COLUMNS).copy()
    for col in X.columns:
        if X[col].dtype == bool or str(X[col].dtype) == "boolean":
            X[col] = X[col].astype("int8")
    return X, frame["TARGET"].astype(int)
