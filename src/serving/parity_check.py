"""
Training/serving parity check: for a random sample of applicants, the features
the OFFLINE store gives training must equal the features the ONLINE store gives
the API, including which values are missing. Exits non-zero on any mismatch.

    python -m src.serving.parity_check --n 2000
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def compare_frames(offline: pd.DataFrame, online: pd.DataFrame, rtol: float = 1e-6, atol: float = 1e-9) -> dict:
    """Both frames: one row per SK_ID_CURR, one column per feature."""
    offline = offline.sort_index().astype("float64")
    online = online.reindex(offline.index)[offline.columns].astype("float64")
    null_mismatch = (offline.isna() != online.isna()).sum()
    both = offline.notna() & online.notna()
    diff = (offline.where(both) - online.where(both)).abs()
    bad_values = (diff > (atol + rtol * offline.abs().where(both))).sum()
    bad = sorted(set(null_mismatch[null_mismatch > 0].index) | set(bad_values[bad_values > 0].index))
    return {
        "n_rows": int(len(offline)),
        "n_features": int(offline.shape[1]),
        "max_abs_diff": float(np.nanmax(diff.to_numpy())) if both.to_numpy().any() else 0.0,
        "null_mismatches": int(null_mismatch.sum()),
        "value_mismatches": int(bad_values.sum()),
        "mismatched_features": bad,
        "ok": not bad,
    }


def main(n: int, seed: int) -> int:
    from feast import FeatureStore

    from src.common import config

    store = FeatureStore(repo_path=str(config.FEAST_REPO_DIR))
    fs = store.get_feature_service(config.FEAST_FEATURE_SERVICE)
    sample = pd.read_parquet(config.FEAST_FEATURES_PARQUET, columns=["SK_ID_CURR", "TARGET"])
    sample = sample.sample(min(n, len(sample)), random_state=seed)
    entity_df = sample.assign(event_timestamp=pd.Timestamp.now(tz="UTC"))

    off = store.get_historical_features(entity_df=entity_df, features=fs).to_df().set_index("SK_ID_CURR")
    on = store.get_online_features(
        features=fs, entity_rows=[{"SK_ID_CURR": int(i)} for i in sample["SK_ID_CURR"]]
    ).to_df().set_index("SK_ID_CURR")
    features = [c for c in on.columns if c in off.columns]
    result = compare_frames(off[features], on[features])
    print({k: v for k, v in result.items() if k != "ok"})
    print("PARITY OK" if result["ok"] else "PARITY FAILED")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    sys.exit(main(args.n, args.seed))
