"""Load the registered champion model and the decision threshold that was saved with it."""
from __future__ import annotations

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.common import config


def load_champion():
    """Returns (model, version, threshold). The threshold is read from the MLflow run
    that produced the champion, so the model and its decision rule can never drift apart."""
    mlflow.set_tracking_uri(f"sqlite:///{config.MLFLOW_DB.as_posix()}")
    client = MlflowClient()
    mv = client.get_model_version_by_alias(config.MODEL_NAME, "champion")
    model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@champion")
    threshold = float(client.get_run(mv.run_id).data.params["decision_threshold"])
    return model, mv.version, threshold
