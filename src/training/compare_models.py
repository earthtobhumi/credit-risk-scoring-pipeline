"""
Phase 3 part 1: XGBoost vs LightGBM, with and without class-imbalance weighting.

Selection metric is PR-AUC (average precision): with ~8% defaults it reflects
how well the model ranks the rare class, where ROC-AUC looks flattering.
ROC-AUC is reported as a secondary metric. Everything is scored on the VALID
split only; the holdout split is never loaded into a model here.

Imbalance strategies:
  none        - train on the natural 8% default rate
  pos_weight  - scale_pos_weight = (#negatives / #positives) from the TRAIN split.
                This improves ranking sometimes, but it distorts predicted
                probabilities, so the chosen model is calibrated in part 2.

Usage:
    python -m src.training.compare_models
    python -m src.training.compare_models --refresh     # re-pull from Feast
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common import config
from src.training.data import feature_matrix, load_training_frame
from src.training.splits import load_or_create_splits

SEED = 42
EARLY_STOPPING_ROUNDS = 50


def _make_model(name: str, scale_pos_weight: float, n_estimators: int):
    if name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=n_estimators, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            tree_method="hist", eval_metric="aucpr",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            scale_pos_weight=scale_pos_weight, random_state=SEED, n_jobs=-1,
        )
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=50,
            scale_pos_weight=scale_pos_weight, random_state=SEED, n_jobs=-1, verbose=-1,
            # LightGBM also monitors binary_logloss by default, and early stopping fires
            # when ANY monitored metric stalls. With scale_pos_weight the predictions are
            # deliberately miscalibrated, so validation logloss is "best" after tree 1 and
            # training stopped there while PR-AUC was still improving. metric="None" turns
            # the default off so early stopping follows PR-AUC (eval_metric) only.
            metric="None",
        )
    raise ValueError(name)


def fit_eval(name, strategy, X_tr, y_tr, X_va, y_va, n_estimators: int = 2000) -> dict:
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)) if strategy == "pos_weight" else 1.0
    model = _make_model(name, spw, n_estimators)
    t0 = time.time()
    if name == "xgboost":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_iter = int(model.best_iteration) + 1
    else:
        model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        best_iter = int(model.best_iteration_)
    proba = model.predict_proba(X_va)[:, 1]
    return {
        "model": name,
        "imbalance": strategy,
        "pr_auc": average_precision_score(y_va, proba),
        "roc_auc": roc_auc_score(y_va, proba),
        "best_iteration": best_iter,
        "fit_seconds": round(time.time() - t0, 1),
    }


def run(refresh: bool = False) -> pd.DataFrame:
    frame = load_training_frame(refresh=refresh)
    X, y = feature_matrix(frame)
    splits = load_or_create_splits(frame["SK_ID_CURR"], y, config.SPLITS_FILE)
    split = frame[["SK_ID_CURR"]].merge(splits, on="SK_ID_CURR", how="left")["split"].to_numpy()

    tr, va = split == "train", split == "valid"
    print(f"rows: train={tr.sum():,}  valid={va.sum():,}  holdout (untouched)={(split == 'holdout').sum():,}")
    print(f"default rate: train={y[tr].mean():.4f}  valid={y[va].mean():.4f}")

    # Reference points, so the model numbers can be judged:
    # a random ranker scores PR-AUC == the default rate; ext_source_mean alone is the
    # strongest single feature in this dataset.
    ext = X.loc[va, "ext_source_mean"].fillna(X.loc[tr, "ext_source_mean"].median())
    print(f"baseline PR-AUC (random ranker) = {y[va].mean():.4f}")
    print(f"baseline ROC-AUC, ext_source_mean alone = {roc_auc_score(y[va], -ext):.4f}")

    rows = []
    for name in ("xgboost", "lightgbm"):
        for strategy in ("none", "pos_weight"):
            print(f"fitting {name} / {strategy} ...", flush=True)
            rows.append(fit_eval(name, strategy, X[tr], y[tr], X[va], y[va]))
    result = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    result.to_csv(config.REPORTS_DIR / "phase3_model_comparison.csv", index=False)
    print("\nValidation results (sorted by PR-AUC):")
    print(result.round(4).to_string(index=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    run(refresh=parser.parse_args().refresh)
