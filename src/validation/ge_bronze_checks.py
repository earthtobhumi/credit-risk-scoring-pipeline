"""
Great Expectations suite: schema + null/range validation applied to Bronze
data before it's allowed to be promoted to Silver.

Design choice (see architecture doc, section 5): we validate here, at the
Bronze->Silver boundary, rather than after PySpark transforms. Catching a
malformed row before it touches feature engineering is cheap; catching it
after it has already fed a trained model is not.
"""
import great_expectations as gx


def build_validator(context, df, suite_name: str = "bronze_applications_suite"):
    datasource = context.sources.add_or_update_pandas("bronze_ds")
    asset = datasource.add_dataframe_asset(name="bronze_asset")
    batch_request = asset.build_batch_request(dataframe=df)

    context.add_or_update_expectation_suite(suite_name)
    validator = context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)

    # --- Schema checks ---
    required_columns = [
        "SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "CODE_GENDER", "DAYS_BIRTH", "DAYS_EMPLOYED",
    ]
    validator.expect_table_columns_to_match_set(
        column_set=None, exact_match=False,
    )
    for col in required_columns:
        validator.expect_column_to_exist(col)

    # --- Identity / uniqueness ---
    validator.expect_column_values_to_not_be_null("SK_ID_CURR")
    validator.expect_column_values_to_be_unique("SK_ID_CURR")

    # --- Null checks on business-critical fields ---
    for col in ["TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "CODE_GENDER"]:
        validator.expect_column_values_to_not_be_null(col)

    # --- Range / domain checks ---
    validator.expect_column_values_to_be_in_set("TARGET", [0, 1])
    validator.expect_column_values_to_be_between("AMT_INCOME_TOTAL", min_value=0, max_value=1e9)
    validator.expect_column_values_to_be_between("AMT_CREDIT", min_value=0, max_value=1e9)
    validator.expect_column_values_to_be_between("AMT_ANNUITY", min_value=0, max_value=1e8)
    validator.expect_column_values_to_be_in_set("CODE_GENDER", ["M", "F"])
    # DAYS_BIRTH / DAYS_EMPLOYED are negative-day-offsets in this schema;
    # DAYS_EMPLOYED has a known sentinel of 365243 for unemployed/retired applicants.
    validator.expect_column_values_to_be_between("DAYS_BIRTH", min_value=-40000, max_value=0)
    validator.expect_column_values_to_be_between(
        "DAYS_EMPLOYED", min_value=-25000, max_value=365243
    )

    validator.save_expectation_suite(discard_failed_expectations=False)
    return validator
