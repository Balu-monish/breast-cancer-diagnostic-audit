"""Data loading with an explicit, tested label contract.

scikit-learn's ``load_breast_cancer`` encodes ``target_names[0] == "malignant"``
(212 samples) and ``target_names[1] == "benign"`` (357 samples) — i.e. raw
``target == 0`` means malignant. Every sklearn metric (``recall_score``,
``precision_score``, ...) defaults to ``pos_label=1``, so consuming the raw
encoding silently computes every "positive class" metric against *benign*,
not malignant. That exact mismatch is what caused the metric-attribution bug
this project exists to fix.

Rather than relying on remembering to pass ``pos_label=0`` correctly at every
call site, this module remaps the label once, at the data boundary, so
``y == 1`` always means malignant (disease present) — the universal clinical
convention "positive = has the condition". ``tests/test_data.py`` pins this
contract down so a future regression would fail loudly instead of silently.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

MALIGNANT = 1
BENIGN = 0


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Wisconsin breast cancer dataset with malignant=1, benign=0.

    Returns:
        X: feature DataFrame, one row per sample, 30 numeric columns.
        y: target Series named "malignant", 1 = malignant, 0 = benign.
    """
    raw = load_breast_cancer(as_frame=True)
    assert raw.target_names[0] == "malignant", raw.target_names
    assert raw.target_names[1] == "benign", raw.target_names

    X = raw.data
    y = (1 - raw.target).rename("malignant")
    return X, y
