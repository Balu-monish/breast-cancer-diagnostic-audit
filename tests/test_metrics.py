import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from bca.metrics import (
    confusion_counts,
    npv,
    ppv,
    sensitivity,
    specificity,
    tune_threshold,
)

# Hand-built: 4 malignant (1), 6 benign (0).
# TP=3 (idx 0,1,2), FN=1 (idx 3), FP=2 (idx 7,8), TN=4 (idx 4,5,6,9)
Y_TRUE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
Y_PRED = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0]


def test_confusion_counts():
    assert confusion_counts(Y_TRUE, Y_PRED) == {"tp": 3, "fn": 1, "fp": 2, "tn": 4}


def test_sensitivity():
    assert sensitivity(Y_TRUE, Y_PRED) == pytest.approx(3 / 4)


def test_specificity():
    assert specificity(Y_TRUE, Y_PRED) == pytest.approx(4 / 6)


def test_ppv():
    assert ppv(Y_TRUE, Y_PRED) == pytest.approx(3 / 5)


def test_npv():
    assert npv(Y_TRUE, Y_PRED) == pytest.approx(4 / 5)


@pytest.fixture(scope="module")
def scored_labels():
    X, y = make_classification(
        n_samples=400, n_features=10, weights=[0.8, 0.2], random_state=42
    )
    clf = LogisticRegression(random_state=42).fit(X, y)
    scores = clf.predict_proba(X)[:, 1]
    return y, scores


def test_tune_threshold_recall_non_increasing_as_precision_floor_rises(scored_labels):
    y_true, y_score = scored_labels
    floors = [0.3, 0.5, 0.7, 0.9]
    recalls = [tune_threshold(y_true, y_score, min_precision=f)["recall"] for f in floors]
    assert all(recalls[i] >= recalls[i + 1] - 1e-9 for i in range(len(recalls) - 1))


def test_tune_threshold_reports_whether_floor_was_met(scored_labels):
    y_true, y_score = scored_labels
    result = tune_threshold(y_true, y_score, min_precision=0.999)
    assert "met_min_precision" in result
    if result["met_min_precision"]:
        assert result["precision"] >= 0.999
