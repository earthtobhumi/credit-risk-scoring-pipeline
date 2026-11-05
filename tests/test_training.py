"""Tests for the Phase 3 split and model-comparison plumbing (no Feast needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.compare_models import fit_eval
from src.training.data import _check_frame, feature_matrix
from src.training.splits import assign_splits, load_or_create_splits


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    return pd.Series(np.arange(n) + 100000), pd.Series(y)


def test_splits_are_disjoint_complete_and_stratified():
    ids, y = _toy()
    s = assign_splits(ids, y)
    assert len(s) == len(ids) and s["SK_ID_CURR"].is_unique
    assert set(s["split"]) == {"train", "valid", "holdout"}
    rate = pd.Series(y.to_numpy(), index=ids.to_numpy()).groupby(s.set_index("SK_ID_CURR")["split"]).mean()
    assert rate.max() - rate.min() < 0.01  # same default rate in every part
    assert abs((s["split"] == "holdout").mean() - 0.15) < 0.01


def test_split_file_is_reused_and_rejects_a_different_population(tmp_path):
    ids, y = _toy()
    path = tmp_path / "splits" / "s.parquet"
    first = load_or_create_splits(ids, y, path)
    again = load_or_create_splits(ids, y, path)
    pd.testing.assert_frame_equal(first, again)
    with pytest.raises(ValueError):
        load_or_create_splits(ids.iloc[:-10], y.iloc[:-10], path)


def test_feature_matrix_casts_bools_and_keeps_nans():
    frame = pd.DataFrame({
        "SK_ID_CURR": [1, 2], "TARGET": [0, 1], "event_timestamp": pd.to_datetime(["2026-01-01"] * 2),
        "is_unemployed_or_retired": [True, False], "ext_source_mean": [np.nan, 0.5],
    })
    X, y = feature_matrix(frame)
    assert list(X.columns) == ["is_unemployed_or_retired", "ext_source_mean"]
    assert X["is_unemployed_or_retired"].tolist() == [1, 0]
    assert np.isnan(X["ext_source_mean"].iloc[0]) and y.tolist() == [0, 1]


def test_feast_frame_check_catches_duplicates_and_expired_features():
    good = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "age_years": [30.0, 40.0, 50.0]})
    _check_frame(good, expected_rows=3)
    with pytest.raises(ValueError):
        _check_frame(pd.concat([good, good.iloc[:1]]), expected_rows=3)  # duplicate row
    with pytest.raises(ValueError):
        _check_frame(good.assign(age_years=np.nan), expected_rows=3)  # expired features


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
@pytest.mark.parametrize("strategy", ["none", "pos_weight"])
def test_fit_eval_learns_a_real_signal(name, strategy):
    rng = np.random.default_rng(1)
    n = 6000
    x1 = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2.5 * x1 - 3)))).astype(int)
    X = pd.DataFrame({"x1": x1, "noise": rng.normal(size=n)})
    X.loc[::7, "noise"] = np.nan  # models must tolerate NaN
    res = fit_eval(name, strategy, X[:4500], pd.Series(y[:4500]), X[4500:], pd.Series(y[4500:]), n_estimators=60)
    assert res["roc_auc"] > 0.8 and res["pr_auc"] > y.mean()


def test_lightgbm_pos_weight_early_stopping_follows_pr_auc_not_logloss():
    # Regression: with scale_pos_weight, LightGBM's default logloss metric stopped
    # training after ONE tree even though PR-AUC kept improving for ~70 trees.
    rng = np.random.default_rng(3)
    n = 40000
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=list("abcdef"))
    z = -2.9 + 0.9 * X["a"] + 0.6 * X["b"] + 0.4 * X["c"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-z))).astype(int))
    res = fit_eval("lightgbm", "pos_weight", X[:30000], y[:30000], X[30000:], y[30000:], n_estimators=300)
    assert res["best_iteration"] > 20
