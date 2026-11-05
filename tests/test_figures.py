import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting.make_figures import main


def test_figures_are_drawn_from_report_files(tmp_path):
    reports, images = tmp_path / "reports", tmp_path / "images"
    reports.mkdir()
    pd.DataFrame({"risk_decile": range(1, 11), "applicants": [4600] * 10,
                  "predicted": [.01, .02, .03, .035, .045, .06, .08, .11, .15, .27],
                  "observed": [.012, .021, .027, .035, .041, .062, .078, .111, .147, .279],
                  "lift_vs_average": [0.1] * 10}).to_csv(reports / "decile_table_holdout.csv", index=False)
    pd.DataFrame({"model": ["lightgbm", "xgboost"], "imbalance": ["none", "pos_weight"], "pr_auc": [.2475, .2453],
                  "roc_auc": [.7544, .7559], "best_iteration": [341, 413], "fit_seconds": [4.1, 11.8]}
                 ).to_csv(reports / "phase3_model_comparison.csv", index=False)
    pd.DataFrame({"feature": ["a", "b", "c"], "mean_abs_shap": [.3, .2, .1], "label": ["A", "B", "C"]}
                 ).to_csv(reports / "shap_importance.csv", index=False)
    assert main(reports, images) == 0
    for name in ("architecture", "risk_bands", "model_comparison", "top_features"):
        assert (images / f"{name}.png").stat().st_size > 5000


def test_missing_report_files_are_skipped_not_fatal(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    assert main(reports, tmp_path / "images") == 0
    assert (tmp_path / "images" / "architecture.png").exists()
    assert not (tmp_path / "images" / "risk_bands.png").exists()
