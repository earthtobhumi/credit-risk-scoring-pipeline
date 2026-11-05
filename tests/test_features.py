"""
Unit tests for Phase 2 feature engineering. Run with: pytest tests/

Deliberately tests `compute_features()` against hand-built Spark rows rather
than the real Bronze/Silver Delta tables: it's faster, and it keeps this
test from touching data/ at all -- see tests/test_ingestion.py's
ingestion tests, which now run against a tmp_path copy of the data
layout and never touch data/.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import Row, SparkSession
from pyspark.sql.types import DoubleType, IntegerType, LongType, StructField, StructType

from src.features.engineer_features import DAYS_EMPLOYED_SENTINEL, compute_features

# Explicit schema so single-row test DataFrames with a None in one column
# don't hit Spark's "CANNOT_DETERMINE_TYPE" schema-inference error (Spark
# can't infer a type from a lone null with no other row to infer it from).
_SCHEMA = StructType(
    [
        StructField("SK_ID_CURR", LongType()),
        StructField("TARGET", IntegerType()),
        StructField("AMT_INCOME_TOTAL", DoubleType()),
        StructField("AMT_CREDIT", DoubleType()),
        StructField("AMT_ANNUITY", DoubleType()),
        StructField("AMT_GOODS_PRICE", DoubleType()),
        StructField("CNT_FAM_MEMBERS", IntegerType()),
        StructField("DAYS_BIRTH", IntegerType()),
        StructField("DAYS_EMPLOYED", IntegerType()),
        StructField("EXT_SOURCE_1", DoubleType()),
        StructField("EXT_SOURCE_2", DoubleType()),
        StructField("EXT_SOURCE_3", DoubleType()),
    ]
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.appName("test-feature-engineering")
        .master("local[1]")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def _base_row(**overrides):
    row = dict(
        SK_ID_CURR=1,
        TARGET=0,
        AMT_INCOME_TOTAL=100_000.0,
        AMT_CREDIT=300_000.0,
        AMT_ANNUITY=15_000.0,
        AMT_GOODS_PRICE=280_000.0,
        CNT_FAM_MEMBERS=2,
        DAYS_BIRTH=-14_600,  # 40 years
        DAYS_EMPLOYED=-3_650,  # 10 years
        EXT_SOURCE_1=0.5,
        EXT_SOURCE_2=0.6,
        EXT_SOURCE_3=0.7,
    )
    row.update(overrides)
    return Row(**row)


def test_dti_and_utilization_ratios(spark):
    df = spark.createDataFrame([_base_row()], schema=_SCHEMA)
    out = compute_features(df).collect()[0]
    assert out["credit_to_income_ratio"] == pytest.approx(3.0)
    assert out["annuity_to_income_ratio"] == pytest.approx(0.15)
    assert out["credit_to_goods_ratio"] == pytest.approx(300_000 / 280_000)
    assert out["income_per_family_member"] == pytest.approx(50_000.0)


def test_age_years_from_days_birth(spark):
    df = spark.createDataFrame([_base_row(DAYS_BIRTH=-14_610)], schema=_SCHEMA)
    out = compute_features(df).collect()[0]
    assert out["age_years"] == pytest.approx(40.0, abs=0.05)


def test_days_employed_sentinel_is_nulled_and_flagged(spark):
    df = spark.createDataFrame([_base_row(DAYS_EMPLOYED=DAYS_EMPLOYED_SENTINEL)], schema=_SCHEMA)
    out = compute_features(df).collect()[0]
    assert out["is_unemployed_or_retired"] is True
    assert out["employment_years"] is None


def test_normal_days_employed_is_not_flagged(spark):
    df = spark.createDataFrame([_base_row(DAYS_EMPLOYED=-3_650)], schema=_SCHEMA)
    out = compute_features(df).collect()[0]
    assert out["is_unemployed_or_retired"] is False
    assert out["employment_years"] == pytest.approx(10.0, abs=0.05)


def test_zero_income_ratio_is_null_not_inf(spark):
    df = spark.createDataFrame([_base_row(AMT_INCOME_TOTAL=0.0)], schema=_SCHEMA)
    out = compute_features(df).collect()[0]
    assert out["credit_to_income_ratio"] is None
    assert out["annuity_to_income_ratio"] is None
    # income_per_family_member divides by CNT_FAM_MEMBERS, not income, so a
    # zero income correctly yields 0.0 here rather than null.
    assert out["income_per_family_member"] == pytest.approx(0.0)


def test_zero_family_members_ratio_is_null_not_inf(spark):
    df = spark.createDataFrame([_base_row(CNT_FAM_MEMBERS=0)], schema=_SCHEMA)
    out = compute_features(df).collect()[0]
    assert out["income_per_family_member"] is None


def test_ext_source_mean_ignores_nulls_not_zeros(spark):
    # Row with one missing EXT_SOURCE: mean should average the other two,
    # not silently treat the missing one as 0.
    df = spark.createDataFrame(
        [_base_row(EXT_SOURCE_1=None, EXT_SOURCE_2=0.8, EXT_SOURCE_3=0.4)], schema=_SCHEMA
    )
    out = compute_features(df).collect()[0]
    assert out["ext_source_mean"] == pytest.approx(0.6)
    assert out["ext_source_missing_count"] == 1


def test_all_ext_sources_missing_gives_null_mean_not_zero(spark):
    df = spark.createDataFrame(
        [_base_row(EXT_SOURCE_1=None, EXT_SOURCE_2=None, EXT_SOURCE_3=None)], schema=_SCHEMA
    )
    out = compute_features(df).collect()[0]
    assert out["ext_source_mean"] is None
    assert out["ext_source_missing_count"] == 3


def test_output_schema_excludes_raw_columns(spark):
    df = spark.createDataFrame([_base_row()], schema=_SCHEMA)
    out_cols = set(compute_features(df).columns)
    assert "SK_ID_CURR" in out_cols and "TARGET" in out_cols
    # Raw upper-case source columns must not leak into Gold; selected raw columns
    # only appear renamed (see application_passthrough.py).
    assert "AMT_INCOME_TOTAL" not in out_cols
