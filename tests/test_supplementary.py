"""Tests for bureau / previous_application validation and aggregate features.

Nothing here touches data/: validation tests use in-memory DataFrames and the
aggregate tests build tiny Spark frames.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import Row, SparkSession

from src.features.supplementary_features import bureau_aggregates, previous_aggregates
from src.validation.supplementary_checks import validate_bureau, validate_previous_application


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.appName("test-supplementary").master("local[1]").getOrCreate()
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def _sdf(spark, rows):
    # Built from Rows, not a pandas frame: createDataFrame(pandas) needs distutils,
    # which Python 3.12 removed (PySpark 3.5 still imports it).
    return spark.createDataFrame([Row(**r) for r in rows])


def _bureau(**kw):
    row = dict(SK_ID_CURR=1, SK_ID_BUREAU=10, CREDIT_ACTIVE="Active", DAYS_CREDIT=-100,
               CREDIT_DAY_OVERDUE=0, AMT_CREDIT_SUM_OVERDUE=0.0, AMT_CREDIT_SUM=1000.0,
               AMT_CREDIT_SUM_DEBT=400.0, CNT_CREDIT_PROLONG=0)
    row.update(kw)
    return row


def _prev(**kw):
    row = dict(SK_ID_CURR=1, SK_ID_PREV=100, NAME_CONTRACT_STATUS="Approved", DAYS_DECISION=-50)
    row.update(kw)
    return row


def test_bureau_validation_flags_each_failure_with_reason():
    df = pd.DataFrame([
        _bureau(SK_ID_BUREAU=1),                          # clean
        _bureau(SK_ID_BUREAU=2, SK_ID_CURR=999),          # orphan
        _bureau(SK_ID_BUREAU=3, CREDIT_ACTIVE="Weird"),   # bad enum
        _bureau(SK_ID_BUREAU=4, DAYS_CREDIT=5),           # positive days
        _bureau(SK_ID_BUREAU=1),                          # duplicate key
    ])
    clean, bad = validate_bureau(df, valid_applicant_ids={1})
    assert len(clean) == 1 and len(bad) == 4
    reasons = bad["_failure_reasons"].tolist()
    assert "orphan_no_silver_application" in reasons[0]
    assert "invalid_credit_active" in reasons[1]
    assert "days_credit_missing_or_positive" in reasons[2]
    assert "null_or_duplicate_sk_id_bureau" in reasons[3]


def test_previous_validation_flags_orphan_and_bad_status():
    df = pd.DataFrame([_prev(), _prev(SK_ID_PREV=101, SK_ID_CURR=2),
                       _prev(SK_ID_PREV=102, NAME_CONTRACT_STATUS="???")])
    clean, bad = validate_previous_application(df, valid_applicant_ids={1})
    assert len(clean) == 1 and len(bad) == 2


def test_bureau_aggregates(spark):
    pdf = [
        _bureau(SK_ID_BUREAU=1, DAYS_CREDIT=-100, CREDIT_DAY_OVERDUE=5, AMT_CREDIT_SUM_OVERDUE=50.0),
        _bureau(SK_ID_BUREAU=2, DAYS_CREDIT=-900, CREDIT_ACTIVE="Closed",
                AMT_CREDIT_SUM=9999.0, AMT_CREDIT_SUM_DEBT=0.0),
    ]
    out = bureau_aggregates(_sdf(spark, pdf)).toPandas().iloc[0]
    assert out["bureau_credit_count"] == 2
    assert out["bureau_active_count"] == 1
    assert out["bureau_max_days_overdue"] == 5
    assert out["bureau_days_since_last_credit"] == 100
    # ratio only uses ACTIVE credits: 400 / 1000, the closed 9999 credit is ignored
    assert out["bureau_active_debt_to_credit_ratio"] == pytest.approx(0.4)


def test_previous_aggregates(spark):
    pdf = [
        _prev(SK_ID_PREV=1, DAYS_DECISION=-10),
        _prev(SK_ID_PREV=2, NAME_CONTRACT_STATUS="Refused", DAYS_DECISION=-400),
        _prev(SK_ID_PREV=3, NAME_CONTRACT_STATUS="Refused", DAYS_DECISION=-800),
        _prev(SK_ID_PREV=4, NAME_CONTRACT_STATUS="Canceled", DAYS_DECISION=-900),
    ]
    out = previous_aggregates(_sdf(spark, pdf)).toPandas().iloc[0]
    assert out["prev_app_count"] == 4
    assert out["prev_approved_count"] == 1
    assert out["prev_refused_count"] == 2
    assert out["prev_refusal_rate"] == pytest.approx(0.5)
    assert out["prev_days_since_last_decision"] == 10


def test_applicant_without_history_gets_zero_counts_and_null_rest(spark):
    from tests.test_features import _SCHEMA, _base_row
    from src.features.engineer_features import compute_features

    apps = spark.createDataFrame([_base_row(SK_ID_CURR=1), _base_row(SK_ID_CURR=2)], schema=_SCHEMA)
    b = bureau_aggregates(_sdf(spark, [_bureau(SK_ID_CURR=1)]))
    p = previous_aggregates(_sdf(spark, [_prev(SK_ID_CURR=1)]))
    out = compute_features(apps, b, p).toPandas().set_index("SK_ID_CURR")
    assert out.loc[2, "bureau_credit_count"] == 0 and out.loc[2, "prev_app_count"] == 0
    assert pd.isna(out.loc[2, "bureau_max_days_overdue"])
    assert pd.isna(out.loc[2, "prev_refusal_rate"])
    assert out.loc[1, "bureau_credit_count"] == 1
