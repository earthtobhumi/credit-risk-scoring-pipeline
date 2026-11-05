"""
Metrics and decision-rule helpers for the credit model. Pure functions of
(y_true, predicted default probability), so they are easy to unit test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Weighted gap between predicted and observed default rate across
    equal-sized probability bins (quantile bins, because defaults are rare and
    equal-width bins would leave almost every row in the first one)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("bin").agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
    return float((g["n"] * (g["obs"] - g["pred"]).abs()).sum() / g["n"].sum())


def core_metrics(y, p) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def operating_point(y, p, threshold: float) -> dict:
    """Decline everyone with predicted default probability >= threshold."""
    y, p = np.asarray(y), np.asarray(p)
    declined = p >= threshold
    approved = ~declined
    n_def = max(int(y.sum()), 1)
    n_good = max(int((y == 0).sum()), 1)
    return {
        "threshold": float(threshold),
        "approval_rate": float(approved.mean()),
        "default_rate_among_approved": float(y[approved].mean()) if approved.any() else float("nan"),
        "defaults_caught": float(((y == 1) & declined).sum() / n_def),
        "good_borrowers_declined": float(((y == 0) & declined).sum() / n_good),
    }


def choose_threshold(y, p, cost_ratio: float) -> float:
    """Threshold minimising expected cost, where approving a defaulter costs
    `cost_ratio` and declining a good borrower costs 1. Searched over the
    observed score quantiles so it is a real score cut-off, not a guess."""
    y, p = np.asarray(y), np.asarray(p)
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 400)))
    costs = [
        cost_ratio * ((y == 1) & (p < t)).sum() + ((y == 0) & (p >= t)).sum() for t in grid
    ]
    return float(grid[int(np.argmin(costs))])


def decile_table(y, p) -> pd.DataFrame:
    """Observed default rate by risk decile (decile 1 = lowest predicted risk)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["risk_decile"] = pd.qcut(df["p"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    out = df.groupby("risk_decile").agg(applicants=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    out["lift_vs_average"] = out["observed"] / df["y"].mean()
    return out.reset_index()
