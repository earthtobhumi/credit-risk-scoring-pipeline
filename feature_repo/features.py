"""
Feast feature definitions for the credit risk pipeline.

Why a plain Parquet FileSource instead of pointing Feast at the Gold Delta
table directly: Feast's local/file offline store reads Parquet paths via
pyarrow dataset discovery. A `deltalake` (delta-rs) table directory is
*also* Parquet under the hood, but it's Parquet data files sitting next to
a `_delta_log/` directory of JSON transaction-log files, plus potentially
tombstoned/orphaned Parquet files from earlier versions -- pyarrow dataset
discovery has no notion of the Delta log and would either choke on the
non-Parquet JSON files or silently include stale, no-longer-live rows.
src/features/engineer_features.py exports a plain, single-snapshot Parquet
file (config.FEAST_FEATURES_PARQUET) specifically for this reason, the same
way it writes the governed table separately via the Spark/Delta bridge.
The Gold Delta table remains the source of truth; this Parquet file is a
derived, Feast-specific view of it, regenerated every time the feature job
runs.

Why every row shares one `feature_timestamp`: this is a single batch
feature computation over a static synthetic snapshot with no natural
per-application event time (no "as-of" date exists in the source schema).
All rows get the timestamp of the feature engineering run itself. This is
fine for offline point-in-time joins (any training-time "as of" timestamp
at or after the run will correctly pull this batch) but means there's
currently no way to time-travel to an *earlier* feature snapshot -- if/when
this pipeline runs on a real schedule against streaming or periodically
refreshed data, each run's export should get its own timestamp (already
true here, since `feature_timestamp` is set to `datetime.now()` at export
time) and historical runs' Parquet snapshots would need to be retained
rather than overwritten, unlike today's overwrite-on-each-run behavior.
"""
from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Bool, Float64, Int64

from src.common import config

application = Entity(
    name="application",
    join_keys=["SK_ID_CURR"],
    description="A single loan application, keyed by Home Credit's SK_ID_CURR.",
)

credit_features_source = FileSource(
    name="credit_features_source",
    path=str(config.FEAST_FEATURES_PARQUET),
    timestamp_field="feature_timestamp",
)

credit_risk_features = FeatureView(
    name="credit_risk_features",
    entities=[application],
    # ttl=0 means "no expiry". feature_timestamp is the time the feature job
    # ran, so with a finite TTL a point-in-time join using entity timestamps
    # later than (run time + TTL) would silently return NULL features.
    # Revisit once the feature job runs on a real schedule.
    ttl=timedelta(seconds=0),
    schema=[
        Field(name="credit_to_income_ratio", dtype=Float64),
        Field(name="annuity_to_income_ratio", dtype=Float64),
        # Proxy for utilization, not true revolving-credit utilization --
        # see engineer_features.py module docstring.
        Field(name="credit_to_goods_ratio", dtype=Float64),
        Field(name="income_per_family_member", dtype=Float64),
        Field(name="age_years", dtype=Float64),
        Field(name="employment_years", dtype=Float64, description="Null for the 365243 sentinel rows"),
        Field(name="is_unemployed_or_retired", dtype=Bool),
        Field(name="ext_source_mean", dtype=Float64),
        Field(name="ext_source_missing_count", dtype=Int64),
        # Raw application columns (see src/features/application_passthrough.py).
        Field(name="ext_source_1", dtype=Float64),
        Field(name="ext_source_2", dtype=Float64),
        Field(name="ext_source_3", dtype=Float64),
        Field(name="days_id_publish", dtype=Float64),
        Field(name="days_registration", dtype=Float64),
        Field(name="days_last_phone_change", dtype=Float64),
        Field(name="goods_price", dtype=Float64),
        Field(name="credit_amount", dtype=Float64),
        Field(name="annuity_amount", dtype=Float64),
        Field(name="income_total", dtype=Float64),
        Field(name="region_rating_client", dtype=Float64),
        Field(name="region_rating_client_w_city", dtype=Float64),
        Field(name="own_car_age", dtype=Float64),
        Field(name="children_count", dtype=Float64),
        Field(name="is_male", dtype=Float64),
        Field(name="education_level", dtype=Float64),
        Field(name="owns_car", dtype=Float64),
        Field(name="owns_realty", dtype=Float64),
        # Bureau (prior credits elsewhere). Counts are 0 for no history; the rest are
        # null for applicants with no bureau rows -- see supplementary_features.py.
        Field(name="bureau_credit_count", dtype=Int64),
        Field(name="bureau_active_count", dtype=Int64),
        Field(name="bureau_max_days_overdue", dtype=Float64),
        Field(name="bureau_total_overdue_amt", dtype=Float64),
        Field(name="bureau_prolong_count", dtype=Float64),
        Field(name="bureau_days_since_last_credit", dtype=Float64),
        Field(name="bureau_active_debt_to_credit_ratio", dtype=Float64),
        # Previous Home Credit applications.
        Field(name="prev_app_count", dtype=Int64),
        Field(name="prev_approved_count", dtype=Int64),
        Field(name="prev_refused_count", dtype=Int64),
        Field(name="prev_refusal_rate", dtype=Float64),
        Field(name="prev_days_since_last_decision", dtype=Float64),
    ],
    online=True,
    source=credit_features_source,
    tags={"phase": "2", "layer": "gold", "data": "home_credit_real"},
)

# Convenience grouping for Phase 3 (modeling) to pull the whole feature set
# by name instead of listing every field again. v2 = v1's 9 application
# features + bureau/previous-application aggregates.
credit_risk_features_v2 = FeatureService(
    name="credit_risk_features_v2",
    features=[credit_risk_features],
)
