import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.batch_score import contributions, top_reasons
from src.training.explain import shap_values_for


def _model():
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"ext_source_mean": rng.random(n), "age_years": rng.uniform(21, 70, n),
                      "bureau_credit_count": rng.integers(0, 9, n).astype(float)})
    X.loc[::9, "age_years"] = np.nan
    y = (rng.random(n) < 1 / (1 + np.exp(-(4 * (0.5 - X["ext_source_mean"]) - 2)))).astype(int)
    return lgb.LGBMClassifier(n_estimators=40, verbose=-1).fit(X, y), X


def test_fast_contributions_match_the_shap_library():
    model, X = _model()
    sample = X.head(300)
    np.testing.assert_allclose(contributions(model, sample), shap_values_for(model, sample), atol=1e-6)


def test_top_reasons_lists_only_risk_raising_features():
    contrib = np.array([[0.5, -0.2, 0.1], [-0.3, -0.1, -0.2], [0.0, 0.9, 0.4]])
    out = top_reasons(contrib, ["ext_source_mean", "age_years", "bureau_credit_count"], k=3)
    assert out[0].tolist() == ["Average external credit score", "Number of past credits elsewhere", ""]
    assert out[1].tolist() == ["", "", ""]  # nothing raises this applicant's risk
    assert out[2][0] == "Age"
