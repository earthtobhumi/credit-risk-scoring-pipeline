"""
Generates synthetic loan-application data that mirrors the schema, dtypes,
and quirks of Kaggle's "Home Credit Default Risk" application_train.csv.

WHY THIS EXISTS:
This build environment cannot reach kaggle.com (network egress is restricted
to package registries). This generator lets you build and test the full
pipeline end to end right now. To use the REAL dataset instead:

    1. Download application_train.csv from:
       https://www.kaggle.com/competitions/home-credit-default-risk/data
    2. Drop it at data/raw/application_train.csv
    3. Re-run this pipeline -- no code changes needed, the schema matches.

Usage:
    python -m src.ingestion.generate_synthetic_data --n-rows 50000
"""
import argparse
import numpy as np
import pandas as pd

from src.common.config import RAW_APPLICATIONS_CSV

RNG = np.random.default_rng(42)


def _sample_categorical(n, categories, weights=None):
    return RNG.choice(categories, size=n, p=weights)


def generate(n_rows: int = 50_000) -> pd.DataFrame:
    n = n_rows

    # TARGET: 1 = defaulted. Real dataset is ~8% positive -- realistically imbalanced.
    target = RNG.choice([0, 1], size=n, p=[0.919, 0.081])

    income = RNG.lognormal(mean=11.8, sigma=0.55, size=n).round(2)  # ~ $130k median
    credit = (income * RNG.uniform(1.5, 6.0, size=n)).round(2)
    annuity = (credit / RNG.uniform(8, 30, size=n)).round(2)
    goods_price = (credit * RNG.uniform(0.85, 1.0, size=n)).round(2)

    # DAYS_BIRTH: negative days relative to application date (real dataset convention)
    age_years = RNG.normal(43, 11, size=n).clip(21, 69)
    days_birth = (-age_years * 365.25).round().astype(int)

    # DAYS_EMPLOYED: negative days employed, but the real dataset has a famous
    # anomaly where unemployed/retired applicants get a sentinel value of 365243.
    employed_years = RNG.exponential(scale=6, size=n).clip(0, 45)
    days_employed = (-employed_years * 365.25).round().astype(int)
    unemployed_mask = RNG.random(n) < 0.18
    days_employed[unemployed_mask] = 365243

    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100_001, 100_001 + n),
        "TARGET": target,
        "NAME_CONTRACT_TYPE": _sample_categorical(n, ["Cash loans", "Revolving loans"], [0.9, 0.1]),
        "CODE_GENDER": _sample_categorical(n, ["M", "F"], [0.34, 0.66]),
        "FLAG_OWN_CAR": _sample_categorical(n, ["Y", "N"], [0.34, 0.66]),
        "FLAG_OWN_REALTY": _sample_categorical(n, ["Y", "N"], [0.69, 0.31]),
        "CNT_CHILDREN": RNG.poisson(0.4, size=n).clip(0, 8),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "NAME_INCOME_TYPE": _sample_categorical(
            n, ["Working", "State servant", "Commercial associate", "Pensioner", "Student"],
            [0.52, 0.08, 0.23, 0.16, 0.01],
        ),
        "NAME_EDUCATION_TYPE": _sample_categorical(
            n, ["Secondary / secondary special", "Higher education", "Incomplete higher",
                "Lower secondary", "Academic degree"],
            [0.71, 0.24, 0.03, 0.015, 0.005],
        ),
        "NAME_FAMILY_STATUS": _sample_categorical(
            n, ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
            [0.64, 0.15, 0.10, 0.06, 0.05],
        ),
        "NAME_HOUSING_TYPE": _sample_categorical(
            n, ["House / apartment", "With parents", "Municipal apartment",
                "Rented apartment", "Office apartment", "Co-op apartment"],
            [0.88, 0.048, 0.036, 0.02, 0.01, 0.006],
        ),
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "OWN_CAR_AGE": np.where(
            RNG.random(n) < 0.66, np.nan, RNG.uniform(0, 25, size=n).round(1)
        ),
        "FLAG_MOBIL": 1,
        "FLAG_EMP_PHONE": _sample_categorical(n, [0, 1], [0.19, 0.81]),
        "FLAG_WORK_PHONE": _sample_categorical(n, [0, 1], [0.80, 0.20]),
        "FLAG_PHONE": _sample_categorical(n, [0, 1], [0.72, 0.28]),
        "FLAG_EMAIL": _sample_categorical(n, [0, 1], [0.91, 0.09]),
        "OCCUPATION_TYPE": _sample_categorical(
            n, ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
                "High skill tech staff", "Accountants", "Medicine staff", None],
            [0.17, 0.10, 0.13, 0.07, 0.07, 0.04, 0.03, 0.03, 0.36],
        ),
        "CNT_FAM_MEMBERS": (RNG.poisson(1.9, size=n) + 1).clip(1, 10),
        "REGION_RATING_CLIENT": _sample_categorical(n, [1, 2, 3], [0.09, 0.71, 0.20]),
        "ORGANIZATION_TYPE": _sample_categorical(
            n, ["Business Entity Type 3", "Self-employed", "Other", "Medicine",
                "Government", "School", "Trade: Type 7", "XNA"],
            [0.22, 0.12, 0.10, 0.06, 0.06, 0.05, 0.04, 0.35],
        ),
        # Home Credit's real external bureau credit score proxies (0-1, higher = safer)
        "EXT_SOURCE_1": RNG.beta(5, 3, size=n).round(4),
        "EXT_SOURCE_2": RNG.beta(5, 3, size=n).round(4),
        "EXT_SOURCE_3": RNG.beta(5, 3, size=n).round(4),
    })

    # Make TARGET weakly correlated with risk signals so downstream modeling
    # (Phase 3) has real signal to learn from, not pure noise.
    risk_score = (
        -0.9 * df["EXT_SOURCE_2"]
        - 0.5 * df["EXT_SOURCE_3"]
        + 0.4 * (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].clip(lower=1))
        - 0.2 * (df["DAYS_EMPLOYED"].clip(upper=0) / -3650)
        + RNG.normal(0, 0.6, size=n)
    )
    threshold = np.quantile(risk_score, 0.919)
    df["TARGET"] = (risk_score > threshold).astype(int)

    # Sprinkle in some realistic data-quality issues for Great Expectations to catch.
    dirty_idx = RNG.choice(n, size=int(n * 0.01), replace=False)
    df.loc[dirty_idx[: len(dirty_idx) // 2], "AMT_INCOME_TOTAL"] = -1  # invalid negative income
    df.loc[dirty_idx[len(dirty_idx) // 2:], "CODE_GENDER"] = "XNA"    # unknown/bad category

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rows", type=int, default=50_000)
    args = parser.parse_args()

    df = generate(args.n_rows)
    df.to_csv(RAW_APPLICATIONS_CSV, index=False)
    print(f"Wrote {len(df):,} rows to {RAW_APPLICATIONS_CSV}")
    print(f"TARGET positive rate: {df['TARGET'].mean():.3%}")
