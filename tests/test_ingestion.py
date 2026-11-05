"""
Smoke tests for the Phase 1 pipeline. Run with: pytest tests/

All reads/writes go to a pytest tmp_path, never the real data/ directories.
The pipeline modules import their paths by name (`from config import X`), so
each module's own copy of the constant is patched, not just config.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.generate_synthetic_data import generate
from src.ingestion import batch_loader
from src.validation import run_validation


@pytest.fixture()
def isolated_data(tmp_path, monkeypatch):
    raw_csv = tmp_path / "raw" / "application_train.csv"
    bronze = str(tmp_path / "bronze" / "applications")
    silver = str(tmp_path / "silver" / "applications")
    quarantine = tmp_path / "quarantine"
    for d in (raw_csv.parent, quarantine):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(batch_loader, "BRONZE_APPLICATIONS_TABLE", bronze)
    monkeypatch.setattr(run_validation, "BRONZE_APPLICATIONS_TABLE", bronze)
    monkeypatch.setattr(run_validation, "SILVER_APPLICATIONS_TABLE", silver)
    monkeypatch.setattr(run_validation, "QUARANTINE_DIR", quarantine)
    return raw_csv


def test_generate_schema_and_target_rate():
    df = generate(n_rows=2000)
    assert "TARGET" in df.columns
    assert "SK_ID_CURR" in df.columns
    assert df["SK_ID_CURR"].is_unique
    assert set(df["TARGET"].unique()) <= {0, 1}
    assert 0.02 < df["TARGET"].mean() < 0.20


def test_batch_load_and_validation_pipeline(isolated_data):
    df = generate(n_rows=3000)
    df.to_csv(isolated_data, index=False)

    n_loaded = batch_loader.load_csv_to_bronze(csv_path=isolated_data)
    assert n_loaded == 3000
    assert batch_loader.bronze_row_count() == 3000

    summary = run_validation.run()
    assert summary["total_rows"] == 3000
    assert summary["passed"] + summary["quarantined"] == summary["total_rows"]
    assert summary["quarantined"] > 0
    assert summary["passed"] > 0
