"""
Features for an INVENTED applicant (the demo page's "made-up applicant" mode).

The page asks for a handful of raw values; the derived features are computed here with the
same formulas as src/features/engineer_features.py (ratios are NULL when the denominator is
not positive). Everything the page does not ask for stays unknown (NaN), which both LightGBM
and the training data treat as "missing", so the model scores it without any real record.
tests/test_serving.py checks these formulas against hand-worked examples.
"""
from __future__ import annotations

NAN = float("nan")
RAW_INPUTS = ("ext_source_1", "ext_source_2", "ext_source_3", "age_years", "employment_years", "income_total",
              "credit_amount", "goods_price", "annuity_amount", "education_level")


def _ratio(numerator, denominator) -> float:
    if numerator is None or denominator is None or denominator <= 0:
        return NAN
    return float(numerator) / float(denominator)


def derive_features(inputs: dict) -> dict:
    """Map the raw page inputs (None = left blank) to model feature values."""
    feats = {k: (NAN if inputs.get(k) is None else float(inputs[k])) for k in RAW_INPUTS}
    scores = [inputs.get(k) for k in ("ext_source_1", "ext_source_2", "ext_source_3")]
    known = [float(v) for v in scores if v is not None]
    feats["ext_source_mean"] = sum(known) / len(known) if known else NAN
    feats["ext_source_missing_count"] = float(3 - len(known))
    feats["credit_to_income_ratio"] = _ratio(inputs.get("credit_amount"), inputs.get("income_total"))
    feats["annuity_to_income_ratio"] = _ratio(inputs.get("annuity_amount"), inputs.get("income_total"))
    feats["credit_to_goods_ratio"] = _ratio(inputs.get("credit_amount"), inputs.get("goods_price"))
    # someone who states years in current employment is, by definition, not in the unemployed/retired group
    feats["is_unemployed_or_retired"] = 0.0 if inputs.get("employment_years") is not None else NAN
    return feats
