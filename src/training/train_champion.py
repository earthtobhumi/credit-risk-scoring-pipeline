"""
Phase 3 part 2: train the chosen model, track it in MLflow, evaluate the
holdout set ONCE, set the decision threshold, explain it, and register it.

Champion (from compare_models, selected on validation PR-AUC): LightGBM, no
class weighting. The four candidates were within 0.003 PR-AUC, so ties were
broken toward the simplest setup, which also keeps probabilities undistorted.

Usage:
    python -m src.training.train_champion
    python -m src.training.train_champion --reeval-holdout   # deliberately re-score the holdout
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.common import config  # noqa: E402
from src.training.compare_models import EARLY_STOPPING_ROUNDS, SEED, _make_model  # noqa: E402
from src.training.data import feature_matrix, load_training_frame  # noqa: E402
from src.training.evaluation import (  # noqa: E402
    choose_threshold, core_metrics, decile_table, operating_point,
)
from src.training.explain import global_importance, reason_codes, shap_values_for  # noqa: E402
from src.training.model_card import build_model_card  # noqa: E402
from src.training.splits import load_or_create_splits  # noqa: E402


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_experiment_id() -> str:
    mlflow.set_tracking_uri(f"sqlite:///{config.MLFLOW_DB.as_posix()}")
    exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if exp is not None:
        return exp.experiment_id
    config.MLFLOW_ARTIFACTS.mkdir(exist_ok=True)
    return mlflow.create_experiment(config.MLFLOW_EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACTS.as_uri())


def run(reeval_holdout: bool = False) -> dict:
    config.REPORTS_DIR.mkdir(exist_ok=True)

    frame = load_training_frame()
    X, y = feature_matrix(frame)
    X = X.drop(columns=[c for c in config.EXCLUDED_FEATURES if c in X.columns]).astype("float64")
    splits = load_or_create_splits(frame["SK_ID_CURR"], y, config.SPLITS_FILE)
    split = frame[["SK_ID_CURR"]].merge(splits, on="SK_ID_CURR", how="left")["split"].to_numpy()
    tr, va, ho = split == "train", split == "valid", split == "holdout"
    sizes = {"train": int(tr.sum()), "valid": int(va.sum()), "holdout": int(ho.sum())}

    # --- fit (train only; early stopping on validation PR-AUC) ---
    model = _make_model("lightgbm", 1.0, 2000)
    model.fit(
        X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    p_va = model.predict_proba(X[va])[:, 1]
    valid = core_metrics(y[va], p_va)
    threshold = choose_threshold(y[va], p_va, config.COST_RATIO_FN_TO_FP)
    op_valid = operating_point(y[va], p_va, threshold)

    # --- explanations on a validation sample (never the holdout) ---
    sample = X[va].sample(5000, random_state=SEED) if va.sum() > 5000 else X[va]
    sv = shap_values_for(model, sample)
    raw = model.predict(sample, raw_score=True)
    gap = float(np.abs(sv.sum(axis=1) + _base_value(model, sample) - raw).max())
    if gap > 1e-3:
        raise RuntimeError(f"SHAP values do not add up to the model output (max gap {gap:.4g}).")
    importance = global_importance(sv, sample.columns)
    importance.to_csv(config.REPORTS_DIR / "shap_importance.csv", index=False)
    shap.summary_plot(sv, sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "shap_summary.png", dpi=130)
    plt.close()
    riskiest = np.argsort(-model.predict_proba(sample)[:, 1])[:3]
    examples = [
        {"predicted_default_probability": float(model.predict_proba(sample.iloc[[i]])[:, 1][0]),
         "reasons": reason_codes(sv[i], sample.iloc[i])}
        for i in riskiest
    ]
    (config.REPORTS_DIR / "example_reason_codes.json").write_text(json.dumps(examples, indent=2))

    # --- holdout: scored once ---
    holdout = op_holdout = prior_holdout = None
    already = config.HOLDOUT_MARKER.exists()
    if already and not reeval_holdout:
        prior_holdout = json.loads(config.HOLDOUT_MARKER.read_text())
    do_holdout = reeval_holdout or not already
    if do_holdout:
        p_ho = model.predict_proba(X[ho])[:, 1]
        holdout = core_metrics(y[ho], p_ho)
        op_holdout = operating_point(y[ho], p_ho, threshold)
        decile_table(y[ho], p_ho).to_csv(config.REPORTS_DIR / "decile_table_holdout.csv", index=False)

    # --- MLflow ---
    exp_id = _get_experiment_id()
    data_sha = _sha256(config.FEAST_FEATURES_PARQUET)
    with mlflow.start_run(experiment_id=exp_id, run_name="lightgbm_no_weighting") as active:
        params = {k: v for k, v in model.get_params().items() if k in (
            "n_estimators", "learning_rate", "num_leaves", "subsample", "colsample_bytree", "min_child_samples", "random_state")}
        mlflow.log_params({**params, "best_iteration": int(model.best_iteration_), "class_weighting": "none",
                           "early_stopping_metric": "average_precision", "cost_ratio_fn_to_fp": config.COST_RATIO_FN_TO_FP,
                           "decision_threshold": round(threshold, 5)})
        mlflow.set_tags({
            "gold_features_sha256": data_sha, "splits_file": config.SPLITS_FILE.name, "n_features": X.shape[1],
            "excluded_features": ",".join(config.EXCLUDED_FEATURES), "feature_list": ",".join(X.columns),
            "python": platform.python_version(), "lightgbm": lgb.__version__, "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__, "holdout_evaluated_in_this_run": str(bool(do_holdout)),
        })
        mlflow.log_metrics({f"valid_{k}": v for k, v in valid.items()})
        mlflow.log_metrics({f"valid_op_{k}": v for k, v in op_valid.items()})
        mlflow.log_metrics({f"split_{k}": v for k, v in sizes.items()})
        if holdout:
            mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout.items()})
            mlflow.log_metrics({f"holdout_op_{k}": v for k, v in op_holdout.items()})

        card_path = config.REPORTS_DIR / "model_card.md"
        signature = infer_signature(X[va].head(100), model.predict_proba(X[va].head(100))[:, 1])
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature,
                                 input_example=X[va].head(5), registered_model_name=config.MODEL_NAME)
        client = MlflowClient()
        version = max(int(v.version) for v in client.search_model_versions(f"name='{config.MODEL_NAME}'"))
        client.set_registered_model_alias(config.MODEL_NAME, "champion", str(version))

        card = build_model_card(
            version=version, run_id=active.info.run_id, n_rows=len(frame), default_rate=float(y.mean()),
            split_sizes=sizes, n_features=X.shape[1], excluded=config.EXCLUDED_FEATURES, valid=valid,
            holdout=holdout, op_valid=op_valid, op_holdout=op_holdout, cost_ratio=config.COST_RATIO_FN_TO_FP,
            top_features=importance, data_sha=data_sha, prior_holdout=prior_holdout,
        )
        card_path.write_text(card, encoding="utf-8")
        for name in ("shap_importance.csv", "shap_summary.png", "example_reason_codes.json", "model_card.md",
                     "decile_table_holdout.csv"):
            f = config.REPORTS_DIR / name
            if f.exists() and (name != "decile_table_holdout.csv" or do_holdout):
                mlflow.log_artifact(str(f))
        if do_holdout:
            config.HOLDOUT_MARKER.write_text(json.dumps(
                {"run_id": active.info.run_id, "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "model_version": version, "metrics": holdout}, indent=2))

    print(f"registered {config.MODEL_NAME} v{version} (alias: champion)  run {active.info.run_id}")
    print(f"best iteration {int(model.best_iteration_)}   features {X.shape[1]}   threshold {threshold:.4f}")
    print("validation :", {k: round(v, 4) for k, v in valid.items()})
    print("holdout    :", {k: round(v, 4) for k, v in holdout.items()} if holdout else "skipped (already evaluated once)")
    op = op_holdout or op_valid
    print("operating point:", {k: round(v, 4) for k, v in op.items()})
    print("top features:", ", ".join(importance["feature"].head(8)))
    return {"valid": valid, "holdout": holdout, "version": version}


def _base_value(model, sample) -> float:
    # shap returns one base value per class (positive class last) for some models and a
    # single value (the positive class) for others; the last element is right in both
    # cases, and run() verifies it with an additivity check.
    return float(np.ravel(shap.TreeExplainer(model).expected_value)[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reeval-holdout", action="store_true")
    run(reeval_holdout=parser.parse_args().reeval_holdout)
