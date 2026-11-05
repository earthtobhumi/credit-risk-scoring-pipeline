"""
Raw application columns passed into Gold alongside the engineered features.

Why: the first Gold layer kept only 21 engineered features, which threw away
most of the signal in the three EXT_SOURCE scores (collapsed into one mean) and
dropped several strong plain columns. Tree models handle raw numeric columns and
NULLs natively, so these go in as they are, renamed to lowercase.

Encodings are deliberately simple and explainable:
  - is_male, owns_car, owns_realty : "M"/"Y" -> 1, "F"/"N" -> 0, anything else NULL
  - education_level                : fixed ordinal order, lowest -> highest
No target-dependent encoding is used, so nothing here can leak the label.

A column is included only if it exists in the Silver table, so the transform
still works on the narrower synthetic schema.
"""
from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

# gold name -> Silver column, cast to double
NUMERIC_PASSTHROUGH = {
    "ext_source_1": "EXT_SOURCE_1",
    "ext_source_2": "EXT_SOURCE_2",
    "ext_source_3": "EXT_SOURCE_3",
    "days_id_publish": "DAYS_ID_PUBLISH",
    "days_registration": "DAYS_REGISTRATION",
    "days_last_phone_change": "DAYS_LAST_PHONE_CHANGE",
    "goods_price": "AMT_GOODS_PRICE",
    "credit_amount": "AMT_CREDIT",
    "annuity_amount": "AMT_ANNUITY",
    "income_total": "AMT_INCOME_TOTAL",
    "region_rating_client": "REGION_RATING_CLIENT",
    "region_rating_client_w_city": "REGION_RATING_CLIENT_W_CITY",
    "own_car_age": "OWN_CAR_AGE",
    "children_count": "CNT_CHILDREN",
}

EDUCATION_ORDER = {
    "Lower secondary": 0.0,
    "Secondary / secondary special": 1.0,
    "Incomplete higher": 2.0,
    "Higher education": 3.0,
    "Academic degree": 4.0,
}

APPLICATION_PASSTHROUGH_FEATURES = list(NUMERIC_PASSTHROUGH) + [
    "is_male",
    "education_level",
    "owns_car",
    "owns_realty",
]


def _yes_no(col: str, yes: str, no: str) -> Column:
    return (
        F.when(F.col(col) == F.lit(yes), F.lit(1.0))
        .when(F.col(col) == F.lit(no), F.lit(0.0))
        .otherwise(F.lit(None).cast("double"))
    )


def passthrough_columns(df: DataFrame) -> list[Column]:
    """Aliased Gold columns for every passthrough source present in `df`."""
    have = set(df.columns)
    cols = [F.col(src).cast("double").alias(name) for name, src in NUMERIC_PASSTHROUGH.items() if src in have]
    if "CODE_GENDER" in have:
        cols.append(_yes_no("CODE_GENDER", "M", "F").alias("is_male"))
    if "NAME_EDUCATION_TYPE" in have:
        mapping = F.create_map(*[F.lit(x) for kv in EDUCATION_ORDER.items() for x in kv])
        cols.append(mapping[F.col("NAME_EDUCATION_TYPE")].cast("double").alias("education_level"))
    if "FLAG_OWN_CAR" in have:
        cols.append(_yes_no("FLAG_OWN_CAR", "Y", "N").alias("owns_car"))
    if "FLAG_OWN_REALTY" in have:
        cols.append(_yes_no("FLAG_OWN_REALTY", "Y", "N").alias("owns_realty"))
    return cols
