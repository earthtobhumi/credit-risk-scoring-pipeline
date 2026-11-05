"""
Phase 2: PySpark feature engineering, Silver -> Gold.

Reads the validated Silver Delta table, computes model-ready features, and
writes:
  1. The governed Gold Delta table (config.GOLD_FEATURES_TABLE) -- the
     source of truth, written the same way Bronze/Silver are (via
     `deltalake.write_deltalake`, see src/common/spark_delta_bridge.py for
     why PySpark doesn't write Delta natively here).
  2. A plain Parquet export for Feast (config.FEAST_FEATURES_PARQUET) with an
     added `feature_timestamp` column. Feast's FileSource wants a clean
     Parquet path with a timestamp column, not a Delta table + transaction
     log directory -- see feature_repo/features.py for the full rationale.

Feature groups (see ARCHITECTURE.md Phase 2 section for the design rationale
behind each):
  - DTI ratios: credit_to_income_ratio, annuity_to_income_ratio
  - Credit utilization proxy: credit_to_goods_ratio (see note below -- this
    is a proxy, not true revolving-credit utilization)
  - Age: age_years, derived from DAYS_BIRTH
  - Employment: employment_years, derived from DAYS_EMPLOYED with the
    365243 "unemployed/retired" sentinel handled explicitly (see note below)
  - Household economics: income_per_family_member
  - Bureau score aggregates: ext_source_mean, ext_source_missing_count

Bureau and previous-application aggregates (real Home Credit data only) are
built in supplementary_features.py and joined in by compute_features when the
Silver bureau / previous_application tables exist. See that module for the
"no history" convention and the sentinel caveat.

Note on credit_to_goods_ratio as a utilization proxy: `application_train.csv`
(real or synthetic) has no revolving credit limit / balance columns -- true
"credit utilization" (balance / limit) lives in `bureau.csv` in the real
dataset and isn't available here either. `AMT_CREDIT / AMT_GOODS_PRICE`
(how far the loan amount exceeds the price of the goods it's financing) is
a real, commonly-used ratio in the Home Credit feature set and the closest
available proxy for "how leveraged is this applicant" -- but it is a proxy.
This is called out here and in the Gold column names so Phase 3 modeling
doesn't accidentally document it as true revolving utilization.

Note on the 365243 DAYS_EMPLOYED sentinel: naively converting to years
would turn unemployed/retired applicants into ~1000-year employees. Rather
than dropping the signal, employment_years is set to NULL for sentinel rows
and a separate `is_unemployed_or_retired` boolean flag is added, so a
downstream model can use the flag directly instead of an XGBoost/LightGBM
split trying to rediscover that 365243 is not a real duration.

Usage:
    python -m src.features.engineer_features
"""
from __future__ import annotations

import datetime as dt

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from deltalake import DeltaTable

from src.common import config
from src.common.spark_delta_bridge import read_delta_as_spark, write_spark_df_to_delta
from src.features.application_passthrough import passthrough_columns
from src.features.spark_ops import safe_ratio
from src.features.supplementary_features import (
    BUREAU_COUNT_FEATURES,
    PREVIOUS_COUNT_FEATURES,
    bureau_aggregates,
    previous_aggregates,
)

DAYS_EMPLOYED_SENTINEL = 365243


def get_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("credit-risk-feature-engineering")
        .master("local[*]")
        .getOrCreate()
    )


_safe_ratio = safe_ratio  # kept so existing imports/tests keep working


def compute_features(
    df: DataFrame,
    bureau_agg: DataFrame | None = None,
    previous_agg: DataFrame | None = None,
) -> DataFrame:
    """Pure transform: Silver-shaped DataFrame in, Gold feature DataFrame out.

    `bureau_agg` / `previous_agg` are the per-applicant outputs of
    supplementary_features.py. When given they are LEFT-joined on SK_ID_CURR;
    applicants without history get 0 counts and NULL everywhere else.

    Kept separate from I/O so it's directly unit-testable (see
    tests/test_features.py) without needing Delta tables or a real Spark
    session per assertion.
    """
    df = df.withColumn(
        "credit_to_income_ratio", _safe_ratio(F.col("AMT_CREDIT"), F.col("AMT_INCOME_TOTAL"))
    ).withColumn(
        "annuity_to_income_ratio", _safe_ratio(F.col("AMT_ANNUITY"), F.col("AMT_INCOME_TOTAL"))
    ).withColumn(
        # Proxy for utilization -- see module docstring. Not true revolving
        # credit utilization; no such column exists in this single-table schema.
        "credit_to_goods_ratio", _safe_ratio(F.col("AMT_CREDIT"), F.col("AMT_GOODS_PRICE"))
    ).withColumn(
        "income_per_family_member", _safe_ratio(F.col("AMT_INCOME_TOTAL"), F.col("CNT_FAM_MEMBERS"))
    ).withColumn(
        "age_years", F.round(-F.col("DAYS_BIRTH") / F.lit(365.25), 1)
    ).withColumn(
        "is_unemployed_or_retired", F.col("DAYS_EMPLOYED") == F.lit(DAYS_EMPLOYED_SENTINEL)
    ).withColumn(
        "employment_years",
        F.when(
            F.col("DAYS_EMPLOYED") == F.lit(DAYS_EMPLOYED_SENTINEL), F.lit(None).cast("double")
        ).otherwise(F.round(-F.col("DAYS_EMPLOYED") / F.lit(365.25), 1)),
    ).withColumn(
        # Mean of the available scores; NULL when all three are missing so a
        # genuinely absent score is not mistaken for a genuinely low one
        # (ext_source_missing_count carries the "how many were missing" signal).
        "ext_source_mean",
        F.when(
            F.coalesce(F.col("EXT_SOURCE_1"), F.col("EXT_SOURCE_2"), F.col("EXT_SOURCE_3")).isNull(),
            F.lit(None).cast("double"),
        ).otherwise(
            (
                F.coalesce(F.col("EXT_SOURCE_1"), F.lit(0.0))
                + F.coalesce(F.col("EXT_SOURCE_2"), F.lit(0.0))
                + F.coalesce(F.col("EXT_SOURCE_3"), F.lit(0.0))
            )
            / (
                F.col("EXT_SOURCE_1").isNotNull().cast("int")
                + F.col("EXT_SOURCE_2").isNotNull().cast("int")
                + F.col("EXT_SOURCE_3").isNotNull().cast("int")
            )
        ),
    ).withColumn(
        "ext_source_missing_count",
        (
            F.col("EXT_SOURCE_1").isNull().cast("int")
            + F.col("EXT_SOURCE_2").isNull().cast("int")
            + F.col("EXT_SOURCE_3").isNull().cast("int")
        ),
    )

    feature_cols = [
        "SK_ID_CURR",
        "TARGET",
        "credit_to_income_ratio",
        "annuity_to_income_ratio",
        "credit_to_goods_ratio",
        "income_per_family_member",
        "age_years",
        "employment_years",
        "is_unemployed_or_retired",
        "ext_source_mean",
        "ext_source_missing_count",
    ]
    out = df.select(*feature_cols, *passthrough_columns(df))

    for agg, count_cols in ((bureau_agg, BUREAU_COUNT_FEATURES), (previous_agg, PREVIOUS_COUNT_FEATURES)):
        if agg is not None:
            out = out.join(agg, on="SK_ID_CURR", how="left").fillna(0, subset=count_cols)
    return out


def run() -> dict:
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        silver_df = read_delta_as_spark(spark, config.SILVER_APPLICATIONS_TABLE)

        # Supplementary tables exist only on the real-data path.
        bureau_agg = previous_agg = None
        if DeltaTable.is_deltatable(config.SILVER_BUREAU_TABLE):
            bureau_agg = bureau_aggregates(read_delta_as_spark(spark, config.SILVER_BUREAU_TABLE))
        if DeltaTable.is_deltatable(config.SILVER_PREVIOUS_APPLICATION_TABLE):
            previous_agg = previous_aggregates(
                read_delta_as_spark(spark, config.SILVER_PREVIOUS_APPLICATION_TABLE)
            )
        gold_df = compute_features(silver_df, bureau_agg, previous_agg)

        n_written = write_spark_df_to_delta(gold_df, config.GOLD_FEATURES_TABLE, mode="overwrite")

        # Feast-facing export: same features + a batch feature_timestamp column.
        # All rows share one timestamp because this is a single batch feature
        # computation with no natural per-row event time (see feature_repo/features.py).
        feast_pdf = gold_df.toPandas()
        feast_pdf["feature_timestamp"] = dt.datetime.now(dt.timezone.utc)
        feast_pdf.to_parquet(config.FEAST_FEATURES_PARQUET, index=False)

        return {
            "silver_rows_in": silver_df.count(),
            "gold_rows_written": n_written,
            "gold_table_path": config.GOLD_FEATURES_TABLE,
            "feast_parquet_path": str(config.FEAST_FEATURES_PARQUET),
        }
    finally:
        spark.stop()


if __name__ == "__main__":
    summary = run()
    print("Feature engineering summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
