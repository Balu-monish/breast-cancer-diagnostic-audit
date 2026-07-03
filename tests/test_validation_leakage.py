"""Structural + numeric guards against the leakage found in the original
notebook, where the "best" PCA component count was chosen by directly
maximizing test-set accuracy."""

import inspect

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from bca.validation import split_data, tune


def test_tune_signature_excludes_test_data():
    params = set(inspect.signature(tune).parameters)
    forbidden = {"x_test", "y_test", "test_x", "test_y", "X_test", "y_test"}
    assert not (params & forbidden), (
        "tune() must never accept test-data parameters — this is what makes "
        "test-set-peeking leakage structurally impossible rather than just "
        "discouraged by convention."
    )


def test_scaler_fit_on_train_differs_from_fit_on_full(full_data):
    X, y = full_data
    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler_train_only = StandardScaler().fit(X_train)
    scaler_full = StandardScaler().fit(X)

    transformed_train_only = scaler_train_only.transform(X_test)
    transformed_full = scaler_full.transform(X_test)

    # If these matched, it would mean the train-only scaler learned the same
    # mean/std as one that saw the test set too — i.e. no leakage protection
    # would be observable. They should differ since train != full data.
    assert not np.allclose(transformed_train_only, transformed_full)


def test_split_is_stratified(full_data):
    X, y = full_data
    X_train, X_test, y_train, y_test = split_data(X, y)

    train_rate = y_train.mean()
    test_rate = y_test.mean()
    overall_rate = y.mean()

    assert train_rate == pytest.approx(overall_rate, abs=0.03)
    assert test_rate == pytest.approx(overall_rate, abs=0.03)
