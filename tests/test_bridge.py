"""Regression test: Delta file URIs are percent-encoded, so a project folder
with spaces in its name ("Credit Risk") must still be readable by Spark."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deltalake import write_deltalake
from pyspark.sql import SparkSession

from src.common.spark_delta_bridge import _to_spark_path, read_delta_as_spark


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.appName("test-bridge").master("local[1]").getOrCreate()
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_read_delta_in_folder_with_spaces(spark, tmp_path):
    table = tmp_path / "Credit Risk" / "data" / "silver tbl"
    table.parent.mkdir(parents=True)
    write_deltalake(str(table), pd.DataFrame({"SK_ID_CURR": [1, 2, 3]}))
    assert read_delta_as_spark(spark, str(table)).count() == 3


def test_to_spark_path_handles_each_uri_shape():
    # Windows: delta-rs returns a bare drive-letter path, percent-encoded.
    assert (
        _to_spark_path("C:/Users/someone/My%20Project/data/x.parquet")
        == "C:/Users/someone/My Project/data/x.parquet"
    )
    # Linux/macOS: a file:// URI.
    assert _to_spark_path("file:///home/a%20b/x.parquet") == "/home/a b/x.parquet"
    # Remote storage is passed through untouched.
    assert _to_spark_path("abfss://c@acct.dfs.core.windows.net/t/x.parquet") == (
        "abfss://c@acct.dfs.core.windows.net/t/x.parquet"
    )
