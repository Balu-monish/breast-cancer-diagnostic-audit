"""Explainability at the level clinicians actually care about: the original
30 measured features, not whatever PCA-reduced space a model happens to
operate in internally.

A naive ``shap.TreeExplainer(pipeline.named_steps["model"])`` would explain
the random forest's *PCA components*, not "mean radius" or "worst texture" —
technically correct but clinically useless, since the tuned pipeline's PCA
step means the forest never sees the original features directly. Both
explainers below instead treat the whole fitted pipeline (scaler -> PCA ->
model) as a black box function of the original features, so attributions are
always reported in units a clinician can act on.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def shap_explanation(fitted_pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame):
    """Model-agnostic SHAP values for a fitted pipeline, in original-feature
    space. Used for the random forest model.

    ``X_background`` should be a small sample (~20-30 rows) of training data
    to keep the permutation-based estimator tractable; ``X_explain`` is the
    (typically small) set of rows to produce explanations for.
    """
    background = shap.sample(X_background, min(30, len(X_background)), random_state=42)

    def predict_malignant_proba(X):
        return fitted_pipeline.predict_proba(X)[:, 1]

    explainer = shap.Explainer(predict_malignant_proba, background, algorithm="permutation")
    return explainer(X_explain)


def mlp_permutation_importance(
    fitted_pipeline, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 20
) -> pd.DataFrame:
    """Global feature importance for the MLP via permutation on the held-out
    test set, scored on average precision (consistent with how models were
    tuned in bca.validation.tune). Fast, model-agnostic, and — unlike SHAP's
    KernelExplainer — doesn't require an extra approximate, slow explainer
    for a non-tree model.
    """
    result = permutation_importance(
        fitted_pipeline,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
