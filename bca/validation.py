"""Leakage-free model selection.

The test set is touched exactly once, in ``bca.metrics.evaluate``. That is
enforced structurally here, not just by convention: ``tune`` does not accept
``X_test``/``y_test`` parameters at all, so there is no code path by which
hyperparameter or PCA-component selection could see held-out data (this is
exactly the kind of leakage that let a prior notebook pick its "best" PCA
component count by maximizing test-set accuracy directly).
"""

import inspect

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from bca.models import make_pipeline, param_grid


def split_data(X, y, test_size: float = 0.15, random_state: int = 42):
    """One stratified split. The returned test set must only ever be passed
    to ``bca.metrics.evaluate``, never to ``tune``."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def tune(model_name: str, X_train, y_train, cv_folds: int = 5) -> GridSearchCV:
    """Select hyperparameters (including PCA n_components) via stratified
    CV on the training set only. Scored on average precision (PR-AUC)
    rather than accuracy, since accuracy is a poor selection criterion
    under class imbalance and rewards ignoring the minority (malignant)
    class.
    """
    pipeline = make_pipeline(model_name)
    grid = param_grid(model_name)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    search = GridSearchCV(
        pipeline, grid, cv=cv, scoring="average_precision", n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    return search


_TEST_DATA_PARAM_NAMES = {"x_test", "y_test", "test_x", "test_y"}
assert not _TEST_DATA_PARAM_NAMES & set(inspect.signature(tune).parameters), (
    "tune() must never accept test-data parameters — this is the structural "
    "guard against the PCA-selection leakage found in the original notebook."
)
