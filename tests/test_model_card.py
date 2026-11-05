import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.model_card import build_model_card

M = {"pr_auc": 0.25, "roc_auc": 0.76, "brier": 0.066, "ece": 0.004}
OP = {"threshold": 0.15, "approval_rate": 0.85, "default_rate_among_approved": 0.05,
      "defaults_caught": 0.4, "good_borrowers_declined": 0.1}
TOP = pd.DataFrame({"feature": ["a"], "label": ["Feature A"], "mean_abs_shap": [0.3]})


def _card(**kw):
    base = dict(version=3, run_id="r", n_rows=1000, default_rate=0.08, split_sizes={"train": 1, "valid": 1, "holdout": 1},
                n_features=38, excluded=["is_male"], valid=M, holdout=None, op_valid=OP, op_holdout=None,
                cost_ratio=5.0, top_features=TOP, data_sha="ab" * 32)
    base.update(kw)
    return build_model_card(**base)


def test_card_with_fresh_holdout():
    assert "holdout (evaluated once)" in _card(holdout=M, op_holdout=OP)


def test_card_does_not_claim_a_later_version_was_scored_on_holdout():
    card = _card(prior_holdout={"model_version": 1, "metrics": M})
    assert "scored once, on v1 only" in card and "On the validation set" in card


def test_card_states_excluded_feature_and_cost_assumption():
    card = _card()
    assert "is_male" in card and "assumption, not a measured value" in card
