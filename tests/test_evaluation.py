import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.evaluation import (
    choose_threshold, core_metrics, decile_table, expected_calibration_error, operating_point,
)
from src.training.explain import global_importance, reason_codes


def _calibrated(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 11, n)
    return (rng.random(n) < p).astype(int), p


def test_ece_is_small_when_calibrated_and_large_when_not():
    y, p = _calibrated()
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p * 3, 0, 1)) > 0.05


def test_operating_point_numbers():
    y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    p = np.array([.1, .1, .2, .2, .9, .6, .3, .1, .2, .3])
    op = operating_point(y, p, threshold=0.5)  # declines the 2 rows at .9 and .6
    assert op["approval_rate"] == pytest.approx(0.8)
    assert op["defaults_caught"] == pytest.approx(2 / 3)
    assert op["default_rate_among_approved"] == pytest.approx(1 / 8)
    assert op["good_borrowers_declined"] == 0


def test_threshold_moves_with_cost_ratio():
    y, p = _calibrated()
    t_cheap = choose_threshold(y, p, cost_ratio=1.0)
    t_costly = choose_threshold(y, p, cost_ratio=10.0)
    assert t_costly < t_cheap  # if defaults cost more, decline more people
    # for calibrated scores the optimum is near 1/(1+ratio)
    assert abs(choose_threshold(y, p, 5.0) - 1 / 6) < 0.05


def test_decile_table_is_monotone_for_a_good_model():
    y, p = _calibrated()
    d = decile_table(y, p)
    assert len(d) == 10 and d["observed"].iloc[-1] > 3 * d["observed"].iloc[0]


def test_core_metrics_keys():
    y, p = _calibrated(5000)
    assert set(core_metrics(y, p)) == {"pr_auc", "roc_auc", "brier", "ece"}


def test_reason_codes_pick_top_risk_raisers_only():
    x = pd.Series({"ext_source_2": 0.1, "age_years": 50.0, "income_total": np.nan})
    codes = reason_codes(np.array([0.7, -0.4, 0.2]), x, top_k=3)
    assert [c["feature"] for c in codes] == ["ext_source_2", "income_total"]
    assert "missing" in codes[1]["reason"]
    assert reason_codes(np.array([-0.1, -0.2, -0.3]), x) == []


def test_global_importance_sorted():
    imp = global_importance(np.array([[1.0, -3.0], [1.0, 3.0]]), ["a", "b"])
    assert imp["feature"].tolist() == ["b", "a"]
