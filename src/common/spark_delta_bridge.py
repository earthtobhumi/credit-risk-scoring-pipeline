"""
Bridges PySpark and the `deltalake` (delta-rs) table format.

Why this file exists (resolves the open question flagged at the end of Phase 1):
PySpark's native Delta connector (`delta-spark`) and the `deltalake` pip package
(delta-rs) are two independent implementations of the same open Delta Lake table
format. `delta-spark` resolves its core JARs from Maven Central at Spark-session
startup -- and Maven Central is unreachable in this sandbox (the same root cause
that ruled out delta-spark for Bronze/Silver in Phase 1, see ARCHITECTURE.md 2.1).
So PySpark cannot be pointed at a delta-rs table with `spark.read.format("delta")`
here.

The fix used throughout Phase 2:
  - READ: ask delta-rs for the authoritative list of live Parquet file URIs for a
    given table version (via `DeltaTable.file_uris()`), then hand that exact file
    list to `spark.read.parquet(*files)`. This is deliberately NOT a folder glob --
    Delta tables can contain tombstoned/orphaned Parquet files from prior versions
    that are no longer part of the live table, and only the transaction log knows
    which files are current. Reading via the log keeps Spark and delta-rs looking
    at identical data.
  - WRITE: Spark computes the feature DataFrame, then it's collected to a pandas
    DataFrame and written with `deltalake.write_deltalake(...)`, the same call
    Bronze/Silver already use. This keeps a single writer implementation for every
    Delta table in the repo and keeps Gold consistent with Bronze/Silver.

Row-count ceiling of the write path: `.toPandas()` collects the full feature set
to the driver. Fine at this dataset's scale (tens of thousands of rows fit
comfortably in memory) and consistent with Bronze/Silver's pandas-based writes.
If Gold ever grows past single-node memory, swap this for Spark's native Parquet
writer (`df.write.parquet(...)`) plus a `deltalake.write_deltalake` call over the
resulting files' schema, or move to a real Databricks workspace and use
delta-spark directly per config.py's swap-path notes.

Swap path to real Databricks: none of this file is needed. `delta-spark` resolves
its JARs from Maven Central (reachable outside this sandbox) or a workspace-local
Maven mirror, so `spark.read.format("delta").load(...)` and
`df.write.format("delta").save(...)` work directly against the abfss:// paths
described in config.py. This file becomes dead code at that point.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from deltalake import DeltaTable, write_deltalake
from pyspark.sql import DataFrame, SparkSession


def _to_spark_path(uri: str) -> str:
    """Turn a delta-rs file URI into a path Spark can open.

    delta-rs returns percent-encoded URIs (a space becomes %20), but Spark
    treats %20 literally, so a project folder like "Credit Risk" fails with
    PATH_NOT_FOUND. Local file URIs are decoded to plain paths; remote ones
    (abfss://, s3://) are passed through untouched.
    """
    parsed = urlparse(uri)
    if len(parsed.scheme) == 1:
        # Windows returns "C:/Users/..." with no file:// prefix, which urlparse
        # misreads as a URL scheme called "c". Treat it as a local path.
        return unquote(uri)
    if parsed.scheme in ("", "file"):
        return Path(url2pathname(parsed.path)).as_posix()
    return uri


def read_delta_as_spark(spark: SparkSession, table_path: str) -> DataFrame:
    """Read a delta-rs table into a Spark DataFrame via its live Parquet files."""
    dt = DeltaTable(table_path)
    files = dt.file_uris()
    if not files:
        raise ValueError(f"Delta table at {table_path} has no live data files (version {dt.version()}).")
    return spark.read.parquet(*[_to_spark_path(f) for f in files])


def write_spark_df_to_delta(df: DataFrame, table_path: str, mode: str = "overwrite") -> int:
    """Collect a Spark DataFrame to the driver and write it as a delta-rs table.

    Returns the row count written. `mode="overwrite"` makes this idempotent --
    re-running the Phase 2 job always reflects the current Silver state, matching
    the idempotency convention already used by run_validation.py for Silver.
    """
    pdf = df.toPandas()
    write_deltalake(table_path, pdf, mode=mode, schema_mode="overwrite")
    return len(pdf)
