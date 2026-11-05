# Credit Risk Scoring Pipeline — Full Architecture

## 1. Problem Statement
Predict loan default risk for applicants using a production-shaped ML pipeline — not a notebook exercise. Simulates a real lending institution's risk-scoring system: raw application data → validated, transformed, feature-stored, versioned model → real-time risk score via API.

**Dataset:** Home Credit Default Risk (Kaggle) or Lending Club — free, public, well-documented.

## 2. Architecture Style
- **Medallion lakehouse pattern** (bronze/silver/gold) — not a flat pandas script.
- **Batch-primary with simulated streaming** — Kafka layer demonstrates real-time capability even though the core dataset is batch.
- **Feature-store-centric** — features computed once, reused consistently between training and serving (avoids train/serve skew, a real production concern).
- **Explainability-first serving** — every prediction returns a SHAP-based reason, not just a score.

## 3. Layer-by-Layer Stack

| Layer | Tech | Role |
|---|---|---|
| Ingestion | Python + Kafka (Confluent free tier) | Simulate streaming loan applications; batch historical load |
| Orchestration | Prefect Cloud (free tier) | Schedules/manages pipeline runs, retries, monitoring |
| Storage | Delta Lake on Databricks Community Edition | Bronze (raw) → Silver (cleaned/validated) → Gold (feature-ready) |
| Data Quality | Great Expectations | Schema checks, null/range validation, drift detection between bronze→silver |
| Processing | PySpark | Feature engineering (DTI ratios, payment history aggregates, credit utilization) |
| Feature Store | Feast (SQLite/local backend) | Consistent feature serving for training + inference |
| Modeling | XGBoost / LightGBM | Binary classification — default risk probability |
| Experiment Tracking | MLflow | Params, metrics, model registry, versioning |
| Explainability | SHAP | Per-prediction feature attribution |
| Serving | FastAPI | Real-time REST endpoint: applicant data in → risk score + top SHAP reasons out |
| Dashboard | Power BI (free desktop) | Risk distribution, feature importance, model drift over time |
| CI/CD | GitHub Actions | Lint, test, auto-deploy on push |
| Secrets | `.env` + GitHub Secrets | No credentials in repo |

## 4. Data Flow

1. **Ingest** → Historical CSV batch-loaded to Bronze; Kafka producer simulates live application stream → Bronze.
2. **Validate** → Great Expectations checks run on Bronze before promotion (nulls, type mismatches, out-of-range values flagged/quarantined).
3. **Transform** → PySpark jobs clean + engineer features → Silver → Gold (analytics-ready).
4. **Feature Store** → Gold-layer features registered in Feast; both training and serving pull from the same feature definitions.
5. **Train** → XGBoost/LightGBM trained on Feast-served features; runs logged to MLflow (metrics: AUC-ROC, precision/recall, calibration); best model pushed to MLflow Model Registry.
6. **Explain** → SHAP values computed per prediction, stored alongside scores.
7. **Serve** → FastAPI loads registered model from MLflow → `/predict` endpoint returns risk score + SHAP top-3 reasons.
8. **Monitor** → Power BI dashboard reads from Gold layer + prediction logs for risk distribution, feature drift, and model performance over time.
9. **CI/CD** → GitHub Actions runs tests on PySpark transforms + FastAPI endpoint on every push; auto-deploys on merge to main.

## 5. Key Design Decisions to Be Ready to Defend
- **Why Feast, not just pandas features?** Prevents train/serve skew — same feature logic used at training and inference time, critical in production lending systems where features must be reproducible.
- **Why Great Expectations before Spark transforms?** Catch bad data at the source (bronze) rather than propagating errors downstream into models — cheaper to fix early.
- **Why SHAP over just accuracy metrics?** Regulatory reality: credit decisions legally require explainability (adverse action reasons) — this isn't cosmetic, it reflects real fintech compliance needs.
- **Why MLflow registry over just saving a `.pkl`?** Versioning + rollback capability — mirrors how real teams manage model lifecycle, not just "train once, done."

## 6. Build Order (~6 days)
1. Data ingestion + Delta Lake + Great Expectations (1.5 days)
2. PySpark feature engineering + Feast (1.5 days)
3. Model training + MLflow + SHAP (1.5 days)
4. FastAPI + Power BI + CI/CD + deploy (1.5 days)

## 7. Resume Bullet (draft)
> Built a production-shaped credit risk scoring pipeline using a Delta Lake medallion architecture on Databricks, enforcing data quality via Great Expectations and serving consistent features through Feast; trained and version-controlled XGBoost models via MLflow, with SHAP-based explainability served through FastAPI and monitored via Power BI.

## Implementation notes (added during build)

See `README.md` for the concrete deviations made to run this in a sandboxed
build environment without external network access (Databricks, Kaggle,
Confluent Cloud): `deltalake` (delta-rs) in place of Databricks' Delta
connector, a synthetic Home-Credit-schema dataset in place of the Kaggle
download, and a file-based stream simulation in place of live Kafka. All
three are documented swap points — the pipeline logic, schema, and Delta
table format are identical to what you'd run against the real services.

### Phase 2 additions

Two more sandbox-driven swap points, both documented in `README.md` and in
code docstrings:

- **PySpark ↔ delta-rs bridge** (`src/common/spark_delta_bridge.py`):
  `delta-spark`'s Maven-based JAR resolution hits the same network wall as
  Phase 1's Bronze/Silver Delta choice. PySpark reads Silver via delta-rs's
  transaction-log file list (not a folder glob) and Gold is written back
  through `deltalake.write_deltalake()`, same as Bronze/Silver.
- **Feast Parquet export**: Feast's file-based offline store can't read a
  Delta table directory (Parquet + `_delta_log/` JSON files) directly, so
  `engineer_features.py` exports a plain Parquet snapshot alongside the
  governed Gold Delta table specifically for Feast to read.

Unlike Phase 1's Kafka/Kaggle substitutions, the **Feast local/SQLite
backend itself is not a substitution** — it's what the original architecture
doc specified ("Feast (SQLite/local backend)"). No swap-path note is needed
there; it's already the intended deployment shape for this stage.

One feature group from the original plan is deliberately deferred rather
than faked: **payment history aggregates** need Home Credit's supplementary
`bureau.csv` / `previous_application.csv` tables, which don't exist in the
single-table synthetic schema. See `engineer_features.py`'s docstring for
the reasoning and the extension point if the real Kaggle dataset is used.


### Phase 3-4 additions (built after the original notes)

- **Real data replaced the synthetic generator** for the feature and model stages: bureau and previous-application tables were added as
  per-applicant aggregates (`supplementary_features.py`), and a set of raw application columns passes through Gold unchanged
  (`application_passthrough.py`).
- **Model selection:** XGBoost vs LightGBM, with and without `scale_pos_weight`, compared on validation PR-AUC. All four were within
  0.003; LightGBM without weighting was chosen as the simplest option with undistorted probabilities.
- **Tracking and registry:** MLflow with a SQLite backend. The decision threshold is stored as a run parameter and read back at serving
  time so the model and its decision rule cannot drift apart.
- **Serving:** FastAPI `/score` reads the Feast online store; the same feature service defined for training is used.
- **Monitoring:** PSI per feature and on the model score, against the training split.
- **Resume rule kept:** Kafka stays a file-based simulation and is not listed as a skill; no cloud or Databricks claims.
