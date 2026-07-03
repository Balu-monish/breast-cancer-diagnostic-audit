"""Model factories and hyperparameter grids.

Every model is wrapped in the same ``StandardScaler -> PCA -> estimator``
pipeline and tuned with the same search procedure (see ``bca.validation``),
so comparisons between models are apples-to-apples rather than one model
getting more tuning effort than another.
"""

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("logistic_regression", "random_forest", "mlp")

# PCA component count is tuned like any other hyperparameter — never chosen
# by peeking at test-set performance (see bca.validation.tune).
_PCA_GRID = [5, 10, 15, 20, 30]


def _make_estimator(model_name: str):
    if model_name == "logistic_regression":
        return LogisticRegression(random_state=42, max_iter=5000)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=42)
    if model_name == "mlp":
        return MLPClassifier(random_state=42, max_iter=2000, early_stopping=True)
    raise ValueError(f"Unknown model_name: {model_name!r}")


def make_pipeline(model_name: str) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=42)),
            ("model", _make_estimator(model_name)),
        ]
    )


def param_grid(model_name: str) -> dict:
    grid = {"pca__n_components": _PCA_GRID}

    if model_name == "logistic_regression":
        grid["model__C"] = [0.01, 0.1, 1, 10]
    elif model_name == "random_forest":
        grid["model__n_estimators"] = [100, 200, 300]
        grid["model__max_depth"] = [None, 5, 10]
        grid["model__min_samples_leaf"] = [1, 2, 4]
    elif model_name == "mlp":
        grid["model__hidden_layer_sizes"] = [(64,), (64, 32), (128, 64)]
        grid["model__activation"] = ["relu", "tanh"]
        grid["model__alpha"] = [1e-4, 1e-3, 1e-2]
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    return grid
