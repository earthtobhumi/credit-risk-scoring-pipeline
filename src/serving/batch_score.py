"""
Batch scoring: score every applicant with the registered champion and export the
result for analysts and BI tools.

Writes a Delta table (data/gold/scored_applicants, the governed copy) and a CSV
(reports/scored_applicants.csv, for Excel / Power BI).

Columns: SK_ID_CURR, split (train / valid / holdout), TARGET, default_probability,
decision, risk_band (1 = lowest risk decile ... 10 = highest), top_risk_factor_1..3
(plain-language reasons), model_version, scored_at.

Read `split` before reporting accuracy: scores on `train` rows are optimistic
because the model learned from those rows. Only `holdout` is an honest sample.

Per-row reasons use LightGBM's built-in tree-SHAP contributions (pred_contrib),
which equal the shap library's values for tree models but run on 300k rows in
seconds; tests/test_batch_score.py checks the two agree.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from deltalake import write_deltalake

from src.training.explain import LABELS


def contributions(model, X: pd.DataFrame) -> np.ndarray:
    """(n_rows, n_features) log-odds contributions, without the trailing bias column."""
    return model.booster_.predict(X, pred_contrib=True)[:, :-1]


def top_reasons(contrib: np.ndarray, columns, k: int = 3) -> np.ndarray:
    """(n_rows, k) array of plain-language labels for the k features raising risk the
    most; empty string where fewer than k features raise it."""
    order = np.argsort(-contrib, axis=1)[:, :k]
    values = np.take_along_axis(contrib, order, axis=1)
    labels = np.array([LABELS.get(c, c) for c in columns], dtype=object)[order]
    return np.where(values > 0, labels, "")


def run() -> pd.DataFrame:
    from src.common import config
    from src.serving.registry import load_champion
    from src.training.data import feature_matrix, load_training_frame
    from src.training.splits import load_or_create_splits

    frame = load_training_frame()
    X, y = feature_matrix(frame)
    model, version, threshold = load_champion()
    X = X[list(model.booster_.feature_name())].astype("float64")
    splits = load_or_create_splits(frame["SK_ID_CURR"], y, config.SPLITS_FILE)

    proba = model.predict_proba(X)[:, 1]
    reasons = top_reasons(contributions(model, X), X.columns)
    out = pd.DataFrame({
        "SK_ID_CURR": frame["SK_ID_CURR"].to_numpy(),
        "TARGET": y.to_numpy(),
        "default_probability": proba.round(6),
        "decision": np.where(proba >= threshold, "decline", "approve"),
        "risk_band": pd.qcut(pd.Series(proba).rank(method="first"), 10, labels=range(1, 11)).astype(int).to_numpy(),
        "top_risk_factor_1": reasons[:, 0], "top_risk_factor_2": reasons[:, 1], "top_risk_factor_3": reasons[:, 2],
        "model_version": str(version),
        "scored_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }).merge(splits, on="SK_ID_CURR", how="left")
    out = out[["SK_ID_CURR", "split"] + [c for c in out.columns if c not in ("SK_ID_CURR", "split")]]

    write_deltalake(config.SCORED_TABLE, out, mode="overwrite", schema_mode="overwrite")
    config.REPORTS_DIR.mkdir(exist_ok=True)
    out.to_csv(config.SCORED_CSV, index=False)

    print(f"scored {len(out):,} applicants with model v{version} (threshold {threshold:.4f})")
    print(f"delta table: {config.SCORED_TABLE}")
    print(f"csv:         {config.SCORED_CSV}")
    print("decision mix:", out["decision"].value_counts(normalize=True).round(4).to_dict())
    ho = out[out["split"] == "holdout"]
    print(f"holdout default rate by risk band 1..10: "
          f"{ho.groupby('risk_band')['TARGET'].mean().round(3).tolist()}")
    return out


if __name__ == "__main__":
    run()
