"""
Demonstrates both Feast retrieval paths against the materialized store, so
Phase 3 (training) and Phase 4 (serving) both have a working, verified
pattern to copy instead of re-deriving Feast's API from docs.

  - get_historical_features: point-in-time-correct offline pull, joining an
    entity dataframe (SK_ID_CURR + TARGET + an "as of" timestamp) against the
    feature view. This is what Phase 3's training script should call to
    build its training set -- pulling TARGET from the entity df alongside
    engineered features guarantees the label and features are joined on the
    exact same SK_ID_CURR/timestamp pairing.
  - get_online_features: single low-latency lookup by entity key from the
    SQLite online store, the shape a Phase 4 FastAPI endpoint will call
    per incoming request.

Usage (from the feature_repo directory, after `feast apply` + `feast materialize`):
    PYTHONPATH=.. python -m src.features.feast_demo
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from feast import FeatureStore

from src.common import config


def run_demo(n_sample_ids: int = 5) -> None:
    store = FeatureStore(repo_path=str(config.FEAST_REPO_DIR))

    gold = pd.read_parquet(config.FEAST_FEATURES_PARQUET)
    sample = gold.sample(n_sample_ids, random_state=7)[["SK_ID_CURR", "TARGET"]].reset_index(drop=True)
    sample["event_timestamp"] = dt.datetime.now(dt.timezone.utc)

    # The feature service lists every feature, so this demo (and Phase 3) never
    # has to maintain its own copy of the feature list.
    feature_refs = store.get_feature_service("credit_risk_features_v2")

    print("=== Offline: get_historical_features (training-style pull) ===")
    training_df = store.get_historical_features(
        entity_df=sample, features=feature_refs
    ).to_df()
    print(training_df.to_string(index=False))

    print("\n=== Online: get_online_features (serving-style single lookup) ===")
    entity_rows = [{"SK_ID_CURR": int(sk)} for sk in sample["SK_ID_CURR"]]
    online_resp = store.get_online_features(features=feature_refs, entity_rows=entity_rows).to_df()
    print(online_resp.to_string(index=False))


if __name__ == "__main__":
    run_demo()
