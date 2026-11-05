"""Tests for the raw-column passthrough into Gold, and its Feast schema."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import Row, SparkSession

from src.features.application_passthrough import (
    APPLICATION_PASSTHROUGH_FEATURES,
    passthrough_columns,
)
from src.features.supplementary_features import BUREAU_FEATURES, PREVIOUS_FEATURES


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.appName("test-passthrough").master("local[1]").getOrCreate()
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_encodings(spark):
    df = spark.createDataFrame([
        Row(CODE_GENDER="M", NAME_EDUCATION_TYPE="Higher education", FLAG_OWN_CAR="Y", FLAG_OWN_REALTY="N", EXT_SOURCE_1=0.3),
        Row(CODE_GENDER="F", NAME_EDUCATION_TYPE="Lower secondary", FLAG_OWN_CAR="N", FLAG_OWN_REALTY="Y", EXT_SOURCE_1=None),
        Row(CODE_GENDER="XNA", NAME_EDUCATION_TYPE="Something new", FLAG_OWN_CAR="?", FLAG_OWN_REALTY="Y", EXT_SOURCE_1=0.9),
    ])
    out = df.select(*passthrough_columns(df)).collect()
    assert [r.is_male for r in out] == [1.0, 0.0, None]
    assert [r.education_level for r in out] == [3.0, 0.0, None]  # unknown category -> NULL, not a guess
    assert [r.owns_car for r in out] == [1.0, 0.0, None]
    assert [r.ext_source_1 for r in out] == [0.3, None, 0.9]


def test_columns_missing_from_silver_are_skipped(spark):
    df = spark.createDataFrame([Row(EXT_SOURCE_1=0.5)])
    assert df.select(*passthrough_columns(df)).columns == ["ext_source_1"]


def test_feast_schema_matches_gold_feature_names():
    # Drift guard: every Gold feature added in Python must be declared in Feast.
    from feature_repo.features import credit_risk_features

    declared = {f.name for f in credit_risk_features.schema}
    expected = set(APPLICATION_PASSTHROUGH_FEATURES) | set(BUREAU_FEATURES) | set(PREVIOUS_FEATURES)
    assert expected <= declared
