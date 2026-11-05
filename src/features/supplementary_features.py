"""
Per-applicant aggregates from the supplementary Home Credit tables.

  - bureau: prior credits the applicant held at OTHER institutions
  - previous_application: prior applications at Home Credit itself

Both tables hold one row per prior credit/application and link to the current
application by SK_ID_CURR, so they are grouped to one row per applicant and
LEFT-joined onto the application features.

No leakage: every DAYS_* column in these tables is measured relative to the
current application date (<= 0 means before it), so the aggregates only use
history that existed when the applicant applied.

"No history" convention: an applicant with no bureau (or previous-application)
rows gets 0 for the plain counts, which is literally true, and NULL for every
other aggregate, so "no history" is never confused with "history and zero
overdue". Tree models route NULLs natively.

Not used on purpose: the DAYS_FIRST_DRAWING / DAYS_FIRST_DUE / DAYS_LAST_DUE* /
DAYS_TERMINATION columns of previous_application carry the same 365243
sentinel as DAYS_EMPLOYED (about 60% of rows in the sample for
DAYS_FIRST_DRAWING). If a later feature needs them, null the sentinel first.

active_debt_to_credit_ratio is debt / credit limit summed over ACTIVE bureau
credits of every type. It is the closest thing to utilisation in this dataset
but mixes revolving and instalment credit, so it is not true revolving
utilisation.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.features.spark_ops import safe_ratio

BUREAU_COUNT_FEATURES = ["bureau_credit_count", "bureau_active_count"]
BUREAU_FEATURES = BUREAU_COUNT_FEATURES + [
    "bureau_max_days_overdue",
    "bureau_total_overdue_amt",
    "bureau_prolong_count",
    "bureau_days_since_last_credit",
    "bureau_active_debt_to_credit_ratio",
]
PREVIOUS_COUNT_FEATURES = ["prev_app_count", "prev_approved_count", "prev_refused_count"]
PREVIOUS_FEATURES = PREVIOUS_COUNT_FEATURES + [
    "prev_refusal_rate",
    "prev_days_since_last_decision",
]


def bureau_aggregates(bureau: DataFrame) -> DataFrame:
    """One row per SK_ID_CURR from Silver bureau rows."""
    is_active = F.col("CREDIT_ACTIVE") == F.lit("Active")
    agg = bureau.groupBy("SK_ID_CURR").agg(
        F.count(F.lit(1)).alias("bureau_credit_count"),
        F.sum(is_active.cast("long")).alias("bureau_active_count"),
        F.max("CREDIT_DAY_OVERDUE").cast("double").alias("bureau_max_days_overdue"),
        F.sum("AMT_CREDIT_SUM_OVERDUE").cast("double").alias("bureau_total_overdue_amt"),
        F.sum("CNT_CREDIT_PROLONG").cast("double").alias("bureau_prolong_count"),
        # DAYS_CREDIT is <= 0; its max is the most recent credit.
        (-F.max("DAYS_CREDIT")).cast("double").alias("bureau_days_since_last_credit"),
        F.sum(F.when(is_active, F.col("AMT_CREDIT_SUM_DEBT"))).alias("_active_debt"),
        F.sum(F.when(is_active, F.col("AMT_CREDIT_SUM"))).alias("_active_credit"),
    )
    return agg.withColumn(
        "bureau_active_debt_to_credit_ratio",
        safe_ratio(F.col("_active_debt"), F.col("_active_credit")),
    ).select("SK_ID_CURR", *BUREAU_FEATURES)


def previous_aggregates(previous: DataFrame) -> DataFrame:
    """One row per SK_ID_CURR from Silver previous_application rows."""
    status = F.col("NAME_CONTRACT_STATUS")
    agg = previous.groupBy("SK_ID_CURR").agg(
        F.count(F.lit(1)).alias("prev_app_count"),
        F.sum((status == F.lit("Approved")).cast("long")).alias("prev_approved_count"),
        F.sum((status == F.lit("Refused")).cast("long")).alias("prev_refused_count"),
        (-F.max("DAYS_DECISION")).cast("double").alias("prev_days_since_last_decision"),
    )
    return agg.withColumn(
        "prev_refusal_rate",
        safe_ratio(F.col("prev_refused_count").cast("double"), F.col("prev_app_count").cast("double")),
    ).select("SK_ID_CURR", *PREVIOUS_FEATURES)
