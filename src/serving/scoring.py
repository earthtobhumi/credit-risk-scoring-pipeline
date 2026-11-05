"""
Scoring logic, independent of the web framework so it can be unit tested.

For one applicant: look the features up in the Feast ONLINE store (the same
feature definitions training pulled from the offline store), score them with
the registered model, apply the saved decision threshold, and explain the score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.serving.custom import derive_features
from src.serving.describe import factor_list
from src.training.explain import reason_codes, shap_values_for


class ApplicantNotFound(Exception):
    """No features in the online store for this applicant."""


class ScoringService:
    def __init__(self, model, threshold: float, model_version: str, store, feature_service,
                 explainer=None, top_k: int = 3, reference_scores=None):
        self.model = model
        self.threshold = float(threshold)
        self.model_version = str(model_version)
        self.store = store
        self.feature_service = feature_service
        self.explainer = explainer
        self.top_k = top_k
        # Predicted probabilities of a reference population (the scored applicants), used only to
        # say where an applicant sits (percentile, risk band); optional.
        self.reference = None if reference_scores is None else np.sort(np.asarray(reference_scores, dtype=float))
        # The model, not the feature store, decides which features it uses and in what order.
        self.features = list(model.booster_.feature_name())

    def _fetch_features(self, sk_id_curr: int) -> pd.DataFrame:
        resp = self.store.get_online_features(
            features=self.feature_service, entity_rows=[{"SK_ID_CURR": int(sk_id_curr)}]
        ).to_dict()
        missing = [f for f in self.features if f not in resp]
        if missing:
            raise RuntimeError(f"Feature store response lacks model features: {missing}")
        values = [resp[f][0] for f in self.features]
        if all(v is None for v in values):
            raise ApplicantNotFound(sk_id_curr)
        row = [np.nan if v is None else float(v) for v in values]
        return pd.DataFrame([row], columns=self.features)

    def score(self, sk_id_curr: int) -> dict:
        return self._finish(self._fetch_features(sk_id_curr), int(sk_id_curr), skip_missing=False)

    def score_custom(self, inputs: dict) -> dict:
        """Score an invented applicant from a few raw values; nothing is looked up."""
        feats = derive_features(inputs)
        X = pd.DataFrame([[feats.get(f, np.nan) for f in self.features]], columns=self.features)
        return self._finish(X, None, skip_missing=True)

    def _finish(self, X: pd.DataFrame, sk_id_curr, skip_missing: bool) -> dict:
        probability = float(self.model.predict_proba(X)[:, 1][0])
        sv = shap_values_for(self.model, X, self.explainer)
        context = {}
        if self.reference is not None and len(self.reference):
            share_safer = float(np.searchsorted(self.reference, probability, side="left") / len(self.reference))
            context = {
                "riskier_than_pct": round(share_safer * 100),
                "risk_band": min(10, int(share_safer * 10) + 1),
                "average_default_probability": round(float(self.reference.mean()), 6),
            }
        return {
            **context,
            "factors": factor_list(sv[0], X.iloc[0], skip_missing=skip_missing),
            "sk_id_curr": sk_id_curr,
            "default_probability": round(probability, 6),
            "decision": "decline" if probability >= self.threshold else "approve",
            "threshold": self.threshold,
            "model_version": self.model_version,
            "top_risk_factors": reason_codes(sv[0], X.iloc[0], top_k=self.top_k),
        }

    def info(self) -> dict:
        return {"model_version": self.model_version, "threshold": self.threshold,
                "n_features": len(self.features), "features": self.features}
