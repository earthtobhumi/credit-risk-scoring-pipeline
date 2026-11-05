"""
Bronze -> Silver promotion gate.

Reads the Bronze Delta table, runs the Great Expectations suite from
ge_bronze_checks.py, and splits rows:
  - passing rows  -> written to the Silver Delta table
  - failing rows  -> written to data/quarantine/ (with the failure reason
    attached) for manual review, instead of silently dropping or -- worse --
    silently letting bad data reach the model.

Usage:
    python -m src.validation.run_validation
"""
import datetime as dt

import great_expectations as gx
import pandas as pd
from deltalake import DeltaTable, write_deltalake

from src.common.config import (
    BRONZE_APPLICATIONS_TABLE,
    BRONZE_BUREAU_TABLE,
    BRONZE_PREVIOUS_APPLICATION_TABLE,
    QUARANTINE_DIR,
    SILVER_APPLICATIONS_TABLE,
    SILVER_BUREAU_TABLE,
    SILVER_PREVIOUS_APPLICATION_TABLE,
)
from src.validation.ge_bronze_checks import build_validator
from src.validation.supplementary_checks import validate_bureau, validate_previous_application


def _load_bronze() -> pd.DataFrame:
    dt_table = DeltaTable(BRONZE_APPLICATIONS_TABLE)
    return dt_table.to_pandas()


def run() -> dict:
    df = _load_bronze()
    df = df.reset_index(drop=True)

    context = gx.get_context()
    validator = build_validator(context, df)
    result = validator.validate(result_format={"result_format": "COMPLETE"})

    failing_idx = set()
    failure_reasons = {}
    for r in result.results:
        if not r.success:
            idxs = r.result.get("unexpected_index_list") or []
            exp_type = r.expectation_config.expectation_type
            for i in idxs:
                failing_idx.add(i)
                failure_reasons.setdefault(i, []).append(exp_type)

    clean_df = df.drop(index=list(failing_idx)).reset_index(drop=True)
    bad_df = df.loc[list(failing_idx)].copy()
    if len(bad_df) > 0:
        bad_df["_failure_reasons"] = bad_df.index.map(
            lambda i: ";".join(failure_reasons.get(i, []))
        )

    # Promote clean rows to Silver (overwrite -- Silver always reflects the
    # latest fully-validated view of Bronze; re-running is idempotent).
    if len(clean_df) > 0:
        write_deltalake(SILVER_APPLICATIONS_TABLE, clean_df, mode="overwrite")

    if len(bad_df) > 0:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        quarantine_path = QUARANTINE_DIR / f"quarantined_{ts}.csv"
        bad_df.to_csv(quarantine_path, index=False)

    summary = {
        "total_rows": len(df),
        "passed": len(clean_df),
        "quarantined": len(bad_df),
        "ge_success": result.success,
    }
    return summary


def run_supplementary() -> dict:
    """Validate bureau / previous_application Bronze tables against Silver applications.

    Must run after `run()`: the referential-integrity check uses the Silver
    application IDs. Tables that were never ingested are skipped (the synthetic
    data path has no supplementary tables).
    """
    valid_ids = set(DeltaTable(SILVER_APPLICATIONS_TABLE).to_pandas(columns=["SK_ID_CURR"])["SK_ID_CURR"])
    jobs = {
        "bureau": (BRONZE_BUREAU_TABLE, SILVER_BUREAU_TABLE, validate_bureau),
        "previous_application": (
            BRONZE_PREVIOUS_APPLICATION_TABLE,
            SILVER_PREVIOUS_APPLICATION_TABLE,
            validate_previous_application,
        ),
    }
    summaries = {}
    for name, (bronze_path, silver_path, validate_fn) in jobs.items():
        if not DeltaTable.is_deltatable(bronze_path):
            summaries[name] = {"skipped": "no Bronze table"}
            continue
        df = DeltaTable(bronze_path).to_pandas()
        clean, bad = validate_fn(df, valid_ids)
        write_deltalake(silver_path, clean, mode="overwrite", schema_mode="overwrite")
        if len(bad):
            ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
            bad.to_csv(QUARANTINE_DIR / f"quarantined_{name}_{ts}.csv", index=False)
        reasons = (
            bad["_failure_reasons"].str.split(";").explode().value_counts().to_dict() if len(bad) else {}
        )
        summaries[name] = {
            "total_rows": len(df),
            "passed": len(clean),
            "quarantined": len(bad),
            "reasons": reasons,
        }
    return summaries


if __name__ == "__main__":
    summary = run()
    print("Validation summary (applications):")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("Validation summary (supplementary tables):")
    for name, info in run_supplementary().items():
        print(f"  {name}: {info}")
