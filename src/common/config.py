"""
Central configuration for the credit risk pipeline.
All paths are relative to the project root so the pipeline runs
identically on a laptop, CI runner, or (later) a real Databricks workspace
-- only these paths need to change to point at cloud storage (e.g. abfss://, s3://).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
QUARANTINE_DIR = DATA_DIR / "quarantine"

# Delta table paths (local filesystem now; swap for abfss://.../bronze on Databricks later)
BRONZE_APPLICATIONS_TABLE = str(BRONZE_DIR / "applications")
SILVER_APPLICATIONS_TABLE = str(SILVER_DIR / "applications")
GOLD_FEATURES_TABLE = str(GOLD_DIR / "credit_features")

# Supplementary Home Credit tables (real data only): one row per prior bureau
# credit / prior Home Credit application, linked to applications by SK_ID_CURR.
BRONZE_BUREAU_TABLE = str(BRONZE_DIR / "bureau")
SILVER_BUREAU_TABLE = str(SILVER_DIR / "bureau")
BRONZE_PREVIOUS_APPLICATION_TABLE = str(BRONZE_DIR / "previous_application")
SILVER_PREVIOUS_APPLICATION_TABLE = str(SILVER_DIR / "previous_application")

# Feast reads features from a plain Parquet file (FileSource), not the Delta log directly --
# see src/common/spark_delta_bridge.py for why. This is exported alongside the governed
# Gold Delta table every time the Phase 2 feature job runs.
FEAST_REPO_DIR = PROJECT_ROOT / "feature_repo"
FEAST_FEATURES_PARQUET = GOLD_DIR / "credit_features_feast" / "features.parquet"

# Simulated streaming "topic" (a local file acting as a Kafka topic stand-in).
# See src/ingestion/streaming_producer.py and docker-compose.kafka.yml for the
# real Confluent Kafka swap.
STREAM_TOPIC_FILE = DATA_DIR / "stream_topic.jsonl"

RAW_APPLICATIONS_CSV = RAW_DIR / "application_train.csv"
RAW_BUREAU_CSV = RAW_DIR / "bureau.csv"
RAW_PREVIOUS_APPLICATION_CSV = RAW_DIR / "previous_application.csv"

for d in (RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, QUARANTINE_DIR, FEAST_FEATURES_PARQUET.parent):
    d.mkdir(parents=True, exist_ok=True)

# --- Phase 3: modeling ---
# The split file is IMMUTABLE once written: it pins which applicants are train /
# valid / holdout so the holdout set can never leak into model selection.
SPLITS_FILE = DATA_DIR / "splits" / "split_v1.parquet"
TRAINING_FRAME_CACHE = DATA_DIR / "training" / "training_frame.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
FEAST_FEATURE_SERVICE = "credit_risk_features_v2"

# --- Phase 3 part 2: tracking, registry, decision rule ---
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = PROJECT_ROOT / "mlartifacts"
MLFLOW_EXPERIMENT = "credit_risk"
MODEL_NAME = "credit_risk_model"
HOLDOUT_MARKER = REPORTS_DIR / "holdout_evaluated.json"

# Model inputs deliberately withheld from training (still kept in Gold so a
# fairness audit can use them). Using sex as a credit-scoring input is
# prohibited in many jurisdictions and is hard to defend to a regulator.
EXCLUDED_FEATURES = ["is_male"]

# Illustrative assumption, not a measured number: approving a borrower who then
# defaults costs about 5x what declining a good borrower costs. Change it to
# reflect a real lender's loss-given-default and margin; the threshold follows.
COST_RATIO_FN_TO_FP = 5.0

# --- Phase 4b: batch scoring and monitoring ---
SCORED_TABLE = str(GOLD_DIR / "scored_applicants")
SCORED_CSV = REPORTS_DIR / "scored_applicants.csv"
DRIFT_REPORT = REPORTS_DIR / "drift_report.csv"
