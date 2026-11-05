"""
Batch ingestion: historical CSVs -> Bronze Delta tables.

Bronze = raw, untouched data plus minimal ingestion metadata. No cleaning,
no type coercion, no dropped rows -- that happens in the Silver layer
(src/validation + Phase 2 PySpark transforms) so we always have an
unmodified audit trail of exactly what arrived.

Three Home Credit tables are supported: application_train (one row per
application), bureau (prior credits at other institutions) and
previous_application (prior Home Credit applications). The last two are
optional -- the synthetic generator only produces application_train.

Usage:
    python -m src.ingestion.batch_loader
"""
from __future__ import annotations

import sys
import datetime as dt
from pathlib import Path

import pandas as pd
from deltalake import DeltaTable, write_deltalake

from src.common.config import (
    BRONZE_APPLICATIONS_TABLE,
    BRONZE_BUREAU_TABLE,
    BRONZE_PREVIOUS_APPLICATION_TABLE,
    RAW_APPLICATIONS_CSV,
    RAW_BUREAU_CSV,
    RAW_PREVIOUS_APPLICATION_CSV,
)


def _load_csv(csv_path: Path, table_path: str, source_name: str, mode: str) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. For synthetic data run "
            "`python -m src.ingestion.generate_synthetic_data`; for real data "
            "drop the Kaggle Home Credit CSV at that path."
        )

    df = pd.read_csv(csv_path)

    # Ingestion metadata -- lets Silver/monitoring trace every row back to
    # when and how it arrived, without touching the original columns.
    df["_ingested_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    df["_source"] = source_name

    write_deltalake(
        table_path,
        df,
        mode=mode,
        schema_mode="merge" if mode == "append" else "overwrite",
    )
    return len(df)


def load_csv_to_bronze(
    csv_path=RAW_APPLICATIONS_CSV, source_name: str = "batch_historical", mode: str = "append"
) -> int:
    return _load_csv(csv_path, BRONZE_APPLICATIONS_TABLE, source_name, mode)


def load_bureau_to_bronze(csv_path=RAW_BUREAU_CSV, mode: str = "overwrite") -> int:
    return _load_csv(csv_path, BRONZE_BUREAU_TABLE, "batch_historical", mode)


def load_previous_to_bronze(csv_path=RAW_PREVIOUS_APPLICATION_CSV, mode: str = "overwrite") -> int:
    return _load_csv(csv_path, BRONZE_PREVIOUS_APPLICATION_TABLE, "batch_historical", mode)


def bronze_row_count(table_path: str | None = None) -> int:
    # Resolved at call time (not as a default argument) so tests can redirect the path.
    return DeltaTable(table_path or BRONZE_APPLICATIONS_TABLE).to_pandas().shape[0]


if __name__ == "__main__":
    # The CLI loads each CSV as a full historical snapshot (mode="overwrite"),
    # so re-running it never duplicates rows. Streaming appends use
    # load_csv_to_bronze's default mode="append" instead.
    n = load_csv_to_bronze(mode="overwrite")
    print(f"applications: loaded {n:,} rows -> {BRONZE_APPLICATIONS_TABLE}")
    if RAW_BUREAU_CSV.exists():
        print(f"bureau: loaded {load_bureau_to_bronze():,} rows -> {BRONZE_BUREAU_TABLE}")
    if RAW_PREVIOUS_APPLICATION_CSV.exists():
        print(
            f"previous_application: loaded {load_previous_to_bronze():,} rows "
            f"-> {BRONZE_PREVIOUS_APPLICATION_TABLE}"
        )
    sys.exit(0)
