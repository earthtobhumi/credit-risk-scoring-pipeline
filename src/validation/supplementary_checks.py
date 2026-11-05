"""
Row-level validation for the supplementary Home Credit tables (bureau,
previous_application), applied at the Bronze -> Silver boundary.

Deliberately plain pandas rather than a second Great Expectations suite:
these checks are few, row-level, and need a cross-table lookup (is this
SK_ID_CURR a validated application?) that is simpler to express directly.
Failing rows are returned with a `_failure_reasons` column so they can be
quarantined with an explanation, exactly like the application table.
"""
from __future__ import annotations

import pandas as pd

BUREAU_CREDIT_ACTIVE_VALUES = {"Active", "Closed", "Sold", "Bad debt"}
PREVIOUS_STATUS_VALUES = {"Approved", "Refused", "Canceled", "Unused offer"}


def _split(df: pd.DataFrame, checks: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """checks maps a failure name to a boolean Series that is True where the row FAILS."""
    flags = pd.DataFrame(checks).fillna(False).astype(bool)
    failing = flags.any(axis=1)
    clean = df.loc[~failing].reset_index(drop=True)
    bad = df.loc[failing].copy()
    if len(bad):
        bad["_failure_reasons"] = flags.loc[failing].apply(
            lambda r: ";".join(r.index[r.values]), axis=1
        )
    return clean, bad


def validate_bureau(df: pd.DataFrame, valid_applicant_ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _split(
        df,
        {
            "null_sk_id_curr": df["SK_ID_CURR"].isna(),
            "null_or_duplicate_sk_id_bureau": df["SK_ID_BUREAU"].isna()
            | df["SK_ID_BUREAU"].duplicated(keep="first"),
            # Referential integrity: bureau rows for an application that failed
            # (or never passed) validation must not reach Silver.
            "orphan_no_silver_application": ~df["SK_ID_CURR"].isin(valid_applicant_ids),
            "invalid_credit_active": ~df["CREDIT_ACTIVE"].isin(BUREAU_CREDIT_ACTIVE_VALUES),
            # DAYS_CREDIT is days relative to the application date: always <= 0.
            "days_credit_missing_or_positive": df["DAYS_CREDIT"].isna() | (df["DAYS_CREDIT"] > 0),
            "negative_days_overdue": df["CREDIT_DAY_OVERDUE"] < 0,
            "negative_overdue_amount": df["AMT_CREDIT_SUM_OVERDUE"] < 0,
        },
    )


def validate_previous_application(
    df: pd.DataFrame, valid_applicant_ids
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _split(
        df,
        {
            "null_sk_id_curr": df["SK_ID_CURR"].isna(),
            "null_or_duplicate_sk_id_prev": df["SK_ID_PREV"].isna()
            | df["SK_ID_PREV"].duplicated(keep="first"),
            "orphan_no_silver_application": ~df["SK_ID_CURR"].isin(valid_applicant_ids),
            "invalid_contract_status": ~df["NAME_CONTRACT_STATUS"].isin(PREVIOUS_STATUS_VALUES),
            "days_decision_missing_or_positive": df["DAYS_DECISION"].isna() | (df["DAYS_DECISION"] > 0),
        },
    )
