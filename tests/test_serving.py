"""Serving tests: no Feast, MLflow or network. A tiny real LightGBM model and a fake online store."""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.app import create_app
from src.serving.parity_check import compare_frames
from src.serving.scoring import ApplicantNotFound, ScoringService

FEATURES = ["ext_source_mean", "age_years", "bureau_credit_count"]


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    n = 5000
    X = pd.DataFrame({"ext_source_mean": rng.random(n), "age_years": rng.uniform(21, 70, n),
                      "bureau_credit_count": rng.integers(0, 10, n).astype(float)})
    y = (rng.random(n) < 1 / (1 + np.exp(-(4 * (0.5 - X["ext_source_mean"]) - 2)))).astype(int)
    return lgb.LGBMClassifier(n_estimators=40, verbose=-1).fit(X, y)


class _Resp:
    def __init__(self, d):
        self.d = d

    def to_dict(self):
        return self.d


class FakeStore:
    """Mimics Feast's online response: dict of one-element lists, None for missing values."""
    def __init__(self, rows):
        self.rows = rows

    def get_online_features(self, features, entity_rows):
        row = self.rows.get(entity_rows[0]["SK_ID_CURR"], {})
        # like Feast, return extra columns the model does not use (e.g. is_male) and shuffled order
        return _Resp({"SK_ID_CURR": [entity_rows[0]["SK_ID_CURR"]], "is_male": [row.get("is_male")],
                      **{k: [row.get(k)] for k in reversed(FEATURES)}})


def _service(model, threshold=0.2, reference=None):
    store = FakeStore({
        1: {"ext_source_mean": 0.05, "age_years": 30.0, "bureau_credit_count": 2, "is_male": 1.0},  # risky
        2: {"ext_source_mean": 0.95, "age_years": 50.0, "bureau_credit_count": 0},                  # safe
        3: {"ext_source_mean": None, "age_years": 40.0, "bureau_credit_count": 1},                  # missing value
    })
    return ScoringService(model, threshold, "7", store, feature_service=None, reference_scores=reference)


def test_risky_applicant_is_declined_with_reasons(model):
    out = _service(model).score(1)
    assert out["decision"] == "decline" and out["default_probability"] > 0.2
    assert out["model_version"] == "7" and out["top_risk_factors"]
    assert out["top_risk_factors"][0]["feature"] == "ext_source_mean"


def test_safe_applicant_is_approved(model):
    assert _service(model).score(2)["decision"] == "approve"


def test_threshold_controls_the_decision_not_the_score(model):
    low, high = _service(model, 0.001).score(2), _service(model, 0.999).score(2)
    assert low["default_probability"] == high["default_probability"]
    assert low["decision"] == "decline" and high["decision"] == "approve"


def test_missing_feature_value_is_handled_as_missing(model):
    assert 0 < _service(model).score(3)["default_probability"] < 1


def test_unknown_applicant_raises(model):
    with pytest.raises(ApplicantNotFound):
        _service(model).score(999)


def test_home_page_is_served_and_calls_the_score_endpoint(model):
    with TestClient(create_app(service=_service(model))) as client:
        r = client.get("/")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        assert "Credit risk scorecard" in r.text and "What moved the score" in r.text and 'fetch("/score"' in r.text
        assert "innerHTML" not in r.text  # API text must never be inserted as HTML


def test_http_endpoints(model):
    with TestClient(create_app(service=_service(model))) as client:
        assert client.get("/health").json()["model_version"] == "7"
        assert client.get("/model-info").json()["n_features"] == 3
        ok = client.post("/score", json={"sk_id_curr": 1})
        assert ok.status_code == 200 and ok.json()["decision"] == "decline"
        assert client.post("/score", json={"sk_id_curr": 999}).status_code == 404
        assert client.post("/score", json={"sk_id_curr": "abc"}).status_code == 422
        assert client.post("/score", json={"sk_id_curr": -5}).status_code == 422


def test_parity_compare_passes_on_identical_and_fails_on_drift():
    off = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [0.5, 0.6, 0.7]}, index=[10, 11, 12])
    assert compare_frames(off, off.copy())["ok"]
    changed = off.copy()
    changed.loc[12, "b"] = 0.9
    r = compare_frames(off, changed)
    assert not r["ok"] and r["mismatched_features"] == ["b"]
    nulled = off.copy()
    nulled.loc[10, "a"] = np.nan
    r = compare_frames(off, nulled)
    assert not r["ok"] and r["null_mismatches"] == 1


def test_factors_show_pushes_in_both_directions_with_plain_wording(model):
    out = _service(model).score(1)
    directions = {f["direction"] for f in out["factors"]}
    assert directions == {"raises", "lowers"}
    by = {f["feature"]: f for f in out["factors"]}
    assert by["ext_source_mean"]["direction"] == "raises" and by["ext_source_mean"]["value_text"] == "0.05"
    contributions = [f["contribution"] for f in out["factors"]]
    assert contributions == sorted(contributions, reverse=True)
    assert "risk_band" not in out  # no reference scores supplied


def test_risk_band_and_percentile_come_from_the_reference_population(model):
    reference = np.linspace(0.0, 1.0, 1000)  # a population spread evenly over 0..1
    out = _service(model, reference=reference).score(1)
    expected_share = out["default_probability"]
    assert abs(out["riskier_than_pct"] - round(expected_share * 100)) <= 1
    assert out["risk_band"] == min(10, int(expected_share * 10) + 1)
    assert abs(out["average_default_probability"] - 0.5) < 0.01
    safe = _service(model, reference=reference).score(2)
    assert safe["risk_band"] < out["risk_band"]


def test_describe_value_wording():
    from src.serving.describe import describe_value

    assert describe_value("days_id_publish", -291) == "291 days"
    assert describe_value("employment_years", 3.33) == "3.3 years"
    assert describe_value("annuity_to_income_ratio", 0.25) == "25% of income"
    assert describe_value("owns_car", 0.0) == "No"
    assert describe_value("education_level", 3.0) == "Higher education"
    assert describe_value("ext_source_mean", float("nan")) == "not available"
    assert describe_value("bureau_credit_count", 4.0) == "4"


def test_derived_features_match_the_spark_definitions():
    from src.serving.custom import derive_features

    f = derive_features({"ext_source_1": None, "ext_source_2": 0.6, "ext_source_3": 0.2, "income_total": 100000,
                         "credit_amount": 500000, "goods_price": 400000, "annuity_amount": 25000, "employment_years": 4})
    assert f["ext_source_mean"] == pytest.approx(0.4)          # mean of the scores that exist
    assert f["ext_source_missing_count"] == 1
    assert f["credit_to_income_ratio"] == pytest.approx(5.0)
    assert f["annuity_to_income_ratio"] == pytest.approx(0.25)
    assert f["credit_to_goods_ratio"] == pytest.approx(1.25)
    assert f["is_unemployed_or_retired"] == 0.0
    blank = derive_features({})
    assert np.isnan(blank["ext_source_mean"]) and blank["ext_source_missing_count"] == 3
    assert np.isnan(blank["credit_to_income_ratio"])
    assert np.isnan(derive_features({"credit_amount": 10, "income_total": 0})["credit_to_income_ratio"])  # zero income -> NULL


def test_custom_applicant_is_scored_without_any_lookup(model):
    svc = _service(model)  # the fake store would return nothing for a made-up applicant; it must not be called
    svc.store = None
    low = svc.score_custom({"ext_source_1": 0.1, "ext_source_2": 0.1, "ext_source_3": 0.1, "age_years": 30})
    high = svc.score_custom({"ext_source_1": 0.9, "ext_source_2": 0.9, "ext_source_3": 0.9, "age_years": 30})
    assert low["sk_id_curr"] is None and low["default_probability"] > high["default_probability"]
    assert all(f["value_text"] != "not available" for f in low["factors"])  # unknown facts are not charted


def test_custom_endpoint_validates_ranges(model):
    with TestClient(create_app(service=_service(model))) as client:
        ok = client.post("/score-custom", json={"ext_source_2": 0.4, "age_years": 40})
        assert ok.status_code == 200 and ok.json()["sk_id_curr"] is None
        assert client.post("/score-custom", json={}).status_code == 200  # everything unknown is allowed
        assert client.post("/score-custom", json={"ext_source_1": 2.0}).status_code == 422
        assert client.post("/score-custom", json={"age_years": 5}).status_code == 422
        assert 'fetch(url' in client.get("/").text and "Try a made-up applicant" in client.get("/").text
