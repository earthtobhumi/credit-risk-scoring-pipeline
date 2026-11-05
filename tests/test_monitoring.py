import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitoring.drift import feature_psi, psi, simulate_shift, status


def _sample(n=20000, seed=0, shift=0.0):
    return np.random.default_rng(seed).normal(shift, 1.0, n)


def test_identical_distributions_are_stable():
    assert psi(_sample(seed=0), _sample(seed=1)) < 0.02


def test_shifted_distribution_is_flagged():
    assert psi(_sample(), _sample(seed=1, shift=1.0)) > 0.25


def test_jump_in_missing_share_counts_as_drift():
    ref = pd.Series(_sample())
    cur = ref.copy()
    cur[:8000] = np.nan  # 40% missing now, 0% before
    assert psi(ref, cur) > 0.25


def test_low_cardinality_and_unseen_values():
    ref = pd.Series([0.0, 1.0, 2.0] * 1000)
    assert psi(ref, ref) < 1e-9
    assert psi(ref, pd.Series([0.0, 1.0, 5.0] * 1000)) > 0.25  # value 5 never seen in reference


def test_status_bands():
    assert [status(v) for v in (0.05, 0.15, 0.4)] == ["stable", "moderate", "significant"]


def test_feature_psi_sorted_and_simulated_shift_detected():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"ext_source_2": rng.random(8000), "credit_amount": rng.uniform(1e4, 1e6, 8000),
                       "age_years": rng.uniform(21, 70, 8000)})
    same = feature_psi(df, df.sample(frac=1.0, random_state=1))
    assert (same["status"] == "stable").all()
    shifted = feature_psi(df, simulate_shift(df))
    assert shifted["psi"].is_monotonic_decreasing
    assert set(shifted[shifted["status"] != "stable"]["feature"]) == {"ext_source_2", "credit_amount"}


def test_simulated_shift_leaves_count_columns_alone():
    df = pd.DataFrame({"ext_source_missing_count": [0, 1, 2, 3] * 100, "ext_source_2": np.linspace(0.1, 0.9, 400)})
    assert simulate_shift(df)["ext_source_missing_count"].equals(df["ext_source_missing_count"])
