"""
Data drift check with the Population Stability Index (PSI).

PSI compares how a feature (or the model score) is distributed in a REFERENCE
sample (the training data) against a CURRENT sample:

    PSI = sum over buckets of (current% - reference%) * ln(current% / reference%)

Common industry reading (a convention, not a statistical test):
    < 0.10 stable    0.10 - 0.25 moderate shift    > 0.25 significant shift

Buckets come from the REFERENCE only: quantile bins for continuous features,
one bucket per value for low-cardinality ones, plus a separate bucket for
missing values (a jump in missingness is drift too) and one for values never
seen in the reference.

There is no live traffic in this project, so the CLI compares the training
split against the holdout split (expected: stable) and can compare against any
other Parquet file with `--current`, or against a deliberately shifted copy
with `--simulate-shift` to show what an alarm looks like.

    python -m src.monitoring.drift
    python -m src.monitoring.drift --simulate-shift
    python -m src.monitoring.drift --current some_new_batch.parquet --fail-above 0.25
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

EPS = 1e-4  # floor so an empty bucket does not give ln(0)


def _bucketer(reference: pd.Series, bins: int):
    ref = reference.dropna()
    if ref.nunique() <= bins:
        cats = np.sort(ref.unique())
        n = len(cats) + 2  # + unseen + missing

        def bucket(x: pd.Series) -> np.ndarray:
            idx = np.searchsorted(cats, x.fillna(cats[0]).to_numpy())
            idx = np.clip(idx, 0, len(cats) - 1)
            known = (cats[idx] == x.fillna(np.nan).to_numpy())
            out = np.where(known, idx, len(cats))
            return np.where(x.isna().to_numpy(), len(cats) + 1, out)

        return bucket, n
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1])
    n = len(edges) + 2  # numeric buckets + missing

    def bucket(x: pd.Series) -> np.ndarray:
        idx = np.searchsorted(edges, x.fillna(0).to_numpy(), side="right")
        return np.where(x.isna().to_numpy(), len(edges) + 1, idx)

    return bucket, n


def psi(reference, current, bins: int = 10) -> float:
    ref, cur = pd.Series(reference).astype("float64"), pd.Series(current).astype("float64")
    bucket, n = _bucketer(ref, bins)
    e = np.bincount(bucket(ref), minlength=n) / max(len(ref), 1)
    a = np.bincount(bucket(cur), minlength=n) / max(len(cur), 1)
    e, a = np.maximum(e, EPS), np.maximum(a, EPS)
    return float(((a - e) * np.log(a / e)).sum())


def status(value: float) -> str:
    return "stable" if value < 0.10 else "moderate" if value < 0.25 else "significant"


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame, features=None, bins: int = 10) -> pd.DataFrame:
    features = list(features or reference.columns)
    rows = [{
        "feature": f, "psi": psi(reference[f], current[f], bins),
        "reference_missing": float(reference[f].isna().mean()), "current_missing": float(current[f].isna().mean()),
    } for f in features]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = out["psi"].map(status)
    return out


def simulate_shift(df: pd.DataFrame) -> pd.DataFrame:
    """An illustrative, made-up shift (weaker external scores, larger loans, shorter
    employment) used only to demonstrate what the alarm looks like."""
    out = df.copy()
    # Only continuous columns: scaling a count such as ext_source_missing_count would
    # produce values that never occur, which is a different (and meaningless) kind of shift.
    for col, factor in (("ext_source_1", 0.8), ("ext_source_2", 0.8), ("ext_source_3", 0.8),
                        ("ext_source_mean", 0.8), ("credit_amount", 1.3), ("employment_years", 0.6)):
        if col in out:
            out[col] = out[col] * factor
    return out


def main(current_path: str | None, simulate: bool, fail_above: float) -> int:
    from src.common import config
    from src.serving.registry import load_champion
    from src.training.data import feature_matrix, load_training_frame
    from src.training.splits import load_or_create_splits

    frame = load_training_frame()
    X, y = feature_matrix(frame)
    model, version, _ = load_champion()
    X = X[list(model.booster_.feature_name())].astype("float64")
    splits = load_or_create_splits(frame["SK_ID_CURR"], y, config.SPLITS_FILE)
    split = frame[["SK_ID_CURR"]].merge(splits, on="SK_ID_CURR", how="left")["split"].to_numpy()
    reference = X[split == "train"]

    if current_path:
        current, label = pd.read_parquet(current_path)[reference.columns].astype("float64"), current_path
    elif simulate:
        current, label = simulate_shift(X[split == "holdout"]), "holdout with a simulated shift"
    else:
        current, label = X[split == "holdout"], "holdout"

    report = feature_psi(reference, current)
    score_psi = psi(model.predict_proba(reference)[:, 1], model.predict_proba(current)[:, 1])
    config.REPORTS_DIR.mkdir(exist_ok=True)
    report.to_csv(config.DRIFT_REPORT, index=False)

    print(f"reference: training split ({len(reference):,} rows)   current: {label} ({len(current):,} rows)   model v{version}")
    print(report.head(10).round(4).to_string(index=False))
    counts = report["status"].value_counts().to_dict()
    print(f"features by status: {counts}")
    print(f"model score PSI: {score_psi:.4f} ({status(score_psi)})")
    worst = max(report["psi"].max(), score_psi)
    print("DRIFT ALARM" if worst > fail_above else "NO SIGNIFICANT DRIFT")
    return 1 if worst > fail_above else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", help="Parquet file with the same feature columns")
    parser.add_argument("--simulate-shift", action="store_true")
    parser.add_argument("--fail-above", type=float, default=0.25)
    args = parser.parse_args()
    sys.exit(main(args.current, args.simulate_shift, args.fail_above))
