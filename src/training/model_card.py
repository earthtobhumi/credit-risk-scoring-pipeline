"""Plain-text model card, generated from the numbers of the run it describes."""
from __future__ import annotations


def _row(name: str, m: dict) -> str:
    return (f"| {name} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | {m['brier']:.4f} | {m['ece']:.4f} |")


def build_model_card(
    *, version, run_id, n_rows, default_rate, split_sizes, n_features, excluded,
    valid, holdout, op_valid, op_holdout, cost_ratio, top_features, data_sha, prior_holdout=None,
) -> str:
    if holdout:
        holdout_block = _row("holdout (evaluated once)", holdout)
    elif prior_holdout:
        # The holdout set was spent on an earlier model version; report those numbers
        # for what they are instead of implying this version was scored on it.
        holdout_block = _row(f"holdout (scored once, on v{prior_holdout['model_version']} only)", prior_holdout["metrics"])
    else:
        holdout_block = "| holdout | not evaluated | | | |"
    op = op_holdout or op_valid
    where = "holdout" if op_holdout else "validation"
    top = "\n".join(f"{i + 1}. {r.label} (mean |SHAP| {r.mean_abs_shap:.4f})" for i, r in top_features.head(10).iterrows())
    return f"""# Model card: credit_risk_model v{version}

**Purpose.** Rank loan applicants by probability of payment difficulty (TARGET = 1) from
application data plus prior-credit history. A portfolio/learning project on public data.
It is not a production credit decision system.

**Data.** Kaggle Home Credit Default Risk: application_train ({n_rows:,} applicants,
{default_rate:.2%} defaults), bureau (prior credits elsewhere) and previous_application
(prior Home Credit applications), joined per applicant. Gold features file sha256 `{data_sha[:16]}`.

**Split.** Stratified random: train {split_sizes['train']:,} / validation {split_sizes['valid']:,} /
holdout {split_sizes['holdout']:,}. Validation drives early stopping and the decision
threshold. The holdout set is scored once and never used for selection.

**Model.** LightGBM gradient-boosted trees, no class weighting (weighting did not improve
ranking in the comparison), early stopping on PR-AUC. {n_features} input features.
Withheld from training on purpose: {', '.join(excluded) or 'none'}.

## Performance

| split | PR-AUC | ROC-AUC | Brier | calibration error |
|---|---|---|---|---|
{_row('validation', valid)}
{holdout_block}

A random ranker scores a PR-AUC equal to the default rate ({default_rate:.4f}).

## Decision rule (illustrative)

Decline applicants with predicted default probability >= {op['threshold']:.3f}. The threshold
minimises expected cost on the validation set assuming that approving a borrower who defaults
costs {cost_ratio:g}x as much as declining a good borrower. That ratio is an assumption, not a measured value.
On the {where} set this approves {op['approval_rate']:.1%} of applicants; the default rate among
approved applicants is {op['default_rate_among_approved']:.2%} (vs {default_rate:.2%} overall),
catching {op['defaults_caught']:.1%} of defaulters while declining {op['good_borrowers_declined']:.1%} of good borrowers.

## Most influential features (mean |SHAP|, validation sample)

{top}

## Limitations

- Random split, not a time split: the data has no application dates, so behaviour on a later period is unknown.
- Labels exist only for applicants who were approved and took the loan (selection bias); rejected applicants are unseen.
- The cost ratio behind the threshold is assumed. Calibration is measured, not corrected.
- No fairness audit was run. Age is a model input; sex is excluded from training but kept in Gold for auditing.
- Competition data from a different market and period than any real deployment.

MLflow run: `{run_id}`.
"""
