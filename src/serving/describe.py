"""Plain-language wording for what the model saw, and the signed factor list the demo page draws."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.explain import LABELS

EDUCATION = ["Lower secondary", "Secondary", "Incomplete higher", "Higher education", "Academic degree"]
_DAYS = {"days_id_publish", "days_registration", "days_last_phone_change",
         "bureau_days_since_last_credit", "prev_days_since_last_decision"}
_YEARS = {"employment_years", "age_years", "own_car_age"}
_YESNO = {"is_unemployed_or_retired", "owns_car", "owns_realty", "is_male"}
_MONEY = {"goods_price", "credit_amount", "annuity_amount", "income_total"}
_SCORE = {"ext_source_1", "ext_source_2", "ext_source_3", "ext_source_mean"}


def describe_value(feature: str, value) -> str:
    """The value in words an applicant could read, e.g. -291 days -> '291 days'."""
    if value is None or pd.isna(value):
        return "not available"
    v = float(value)
    if feature in _DAYS:
        return f"{abs(v):,.0f} days"
    if feature in _YEARS:
        return f"{v:.1f} years" if feature != "age_years" else f"{v:.0f} years"
    if feature in _YESNO:
        return "Yes" if v >= 0.5 else "No"
    if feature in _MONEY:
        return f"{v:,.0f}"
    if feature in _SCORE:
        return f"{v:.2f}"
    if feature == "credit_to_income_ratio":
        return f"{v:.1f}x yearly income"
    if feature == "annuity_to_income_ratio":
        return f"{v * 100:.0f}% of income"
    if feature == "credit_to_goods_ratio":
        return f"{v * 100:.0f}% of goods price"
    if feature == "bureau_active_debt_to_credit_ratio":
        return f"{v * 100:.0f}% of limit in use"
    if feature == "prev_refusal_rate":
        return f"{v * 100:.0f}% refused"
    if feature == "education_level":
        return EDUCATION[int(min(max(round(v), 0), len(EDUCATION) - 1))]
    if v == int(v) and abs(v) < 1e9:
        return f"{v:,.0f}"
    return f"{v:.4g}"


def factor_list(shap_row, x_row: pd.Series, k_each: int = 4, skip_missing: bool = False) -> list[dict]:
    """The strongest factors pushing the score up and the strongest pushing it down,
    largest push first. Contributions are in log-odds, relative to a typical applicant."""
    shap_row = np.asarray(shap_row).copy()
    if skip_missing:  # for invented applicants: do not chart facts the user never entered
        shap_row[np.isnan(x_row.to_numpy(dtype=float))] = 0.0
    picks = [i for i in np.argsort(-shap_row)[:k_each] if shap_row[i] > 0]
    picks += [i for i in np.argsort(shap_row)[:k_each] if shap_row[i] < 0]
    picks.sort(key=lambda i: -shap_row[i])
    out = []
    for i in picks:
        name = x_row.index[i]
        out.append({
            "feature": name,
            "label": LABELS.get(name, name),
            "value_text": describe_value(name, x_row.iloc[i]),
            "contribution": float(shap_row[i]),
            "direction": "raises" if shap_row[i] > 0 else "lowers",
        })
    return out
