"""Explainers are fit on small subsets here (not via full bca.validation.tune)
to keep the test suite fast — correctness of the pipeline itself is covered
by test_validation_leakage.py."""

import pytest

from bca.explain import mlp_permutation_importance, shap_explanation
from bca.models import make_pipeline


@pytest.fixture(scope="module")
def fitted_rf(split):
    X_train, X_test, y_train, y_test = split
    pipeline = make_pipeline("random_forest")
    pipeline.set_params(pca__n_components=10, model__n_estimators=50)
    pipeline.fit(X_train, y_train)
    return pipeline


@pytest.fixture(scope="module")
def fitted_mlp(split):
    X_train, X_test, y_train, y_test = split
    pipeline = make_pipeline("mlp")
    pipeline.set_params(pca__n_components=10, model__hidden_layer_sizes=(16,))
    pipeline.fit(X_train, y_train)
    return pipeline


def test_shap_explanation_shape(split, fitted_rf):
    X_train, X_test, y_train, y_test = split
    explanation = shap_explanation(fitted_rf, X_background=X_train, X_explain=X_test.iloc[:5])
    assert explanation.values.shape == (5, X_test.shape[1])


def test_mlp_permutation_importance_shape(split, fitted_mlp):
    X_train, X_test, y_train, y_test = split
    result = mlp_permutation_importance(fitted_mlp, X_test, y_test, n_repeats=3)
    assert list(result.columns) == ["feature", "importance_mean", "importance_std"]
    assert len(result) == X_test.shape[1]
    assert result["importance_mean"].is_monotonic_decreasing
