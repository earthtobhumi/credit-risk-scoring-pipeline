"""
SHAP explanations and plain-language reason codes.

Reason codes answer "why was this applicant scored as risky?" in words a
loan officer or applicant could read: the features pushing the score UP the
most, each with its actual value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = {
    "ext_source_1": "External credit score 1", "ext_source_2": "External credit score 2",
    "ext_source_3": "External credit score 3", "ext_source_mean": "Average external credit score",
    "ext_source_missing_count": "Number of missing external scores",
    "credit_to_income_ratio": "Loan amount relative to income", "annuity_to_income_ratio": "Repayment relative to income",
    "credit_to_goods_ratio": "Loan amount relative to goods price", "income_per_family_member": "Income per family member",
    "age_years": "Age", "employment_years": "Years in current employment",
    "is_unemployed_or_retired": "Unemployed or retired", "days_id_publish": "Days since ID document changed",
    "days_registration": "Days since registration changed", "days_last_phone_change": "Days since phone number changed",
    "goods_price": "Price of goods financed", "credit_amount": "Loan amount", "annuity_amount": "Loan repayment amount",
    "income_total": "Total income", "region_rating_client": "Home region rating",
    "region_rating_client_w_city": "Home region and city rating", "own_car_age": "Age of owned car",
    "children_count": "Number of children", "education_level": "Education level",
    "owns_car": "Owns a car", "owns_realty": "Owns real estate",
    "bureau_credit_count": "Number of past credits elsewhere", "bureau_active_count": "Number of active credits elsewhere",
    "bureau_max_days_overdue": "Longest overdue spell on past credits", "bureau_total_overdue_amt": "Total overdue amount on past credits",
    "bureau_prolong_count": "Number of credit extensions", "bureau_days_since_last_credit": "Days since most recent outside credit",
    "bureau_active_debt_to_credit_ratio": "Debt relative to limit on active credits",
    "prev_app_count": "Number of previous applications here", "prev_approved_count": "Previous applications approved",
    "prev_refused_count": "Previous applications refused", "prev_refusal_rate": "Share of previous applications refused",
    "prev_days_since_last_decision": "Days since last application decision",
}


def shap_values_for(model, X: pd.DataFrame, explainer=None) -> np.ndarray:
    """SHAP values for the positive (default) class as an (n_rows, n_features) array.
    Handles the list-of-two-arrays output some shap/LightGBM versions return.
    Pass a prebuilt `explainer` when scoring repeatedly (the API does)."""
    import shap

    values = (explainer or shap.TreeExplainer(model)).shap_values(X)
    if isinstance(values, list):
        values = values[1]
    return np.asarray(values)


def reason_codes(shap_row, x_row: pd.Series, top_k: int = 3) -> list[dict]:
    """Top features raising this applicant's default risk, largest first.
    Features that lower risk are never listed as reasons."""
    order = np.argsort(-np.asarray(shap_row))
    reasons = []
    for i in order[:top_k]:
        if shap_row[i] <= 0:
            break
        name = x_row.index[i]
        value = x_row.iloc[i]
        reasons.append({
            "feature": name,
            "reason": f"{LABELS.get(name, name)} ({'missing' if pd.isna(value) else f'{value:.4g}'}) raises risk",
            "contribution": float(shap_row[i]),
        })
    return reasons


def global_importance(shap_matrix: np.ndarray, columns) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": list(columns), "mean_abs_shap": np.abs(shap_matrix).mean(axis=0)})
    imp["label"] = imp["feature"].map(LABELS).fillna(imp["feature"])
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
