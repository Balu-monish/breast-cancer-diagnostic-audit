"""These are the exact checks that would have caught the original notebook's
benign/malignant label swap: it stated 357 malignant / 212 benign, backwards
from sklearn's real 212 malignant / 357 benign encoding."""

from sklearn.datasets import load_breast_cancer

from bca.data import load_data


def test_sklearn_raw_encoding_is_malignant_first():
    raw = load_breast_cancer()
    assert raw.target_names[0] == "malignant"
    assert raw.target_names[1] == "benign"


def test_raw_class_counts():
    raw = load_breast_cancer()
    malignant_count = int((raw.target == 0).sum())
    benign_count = int((raw.target == 1).sum())
    assert malignant_count == 212
    assert benign_count == 357


def test_remapped_label_is_named_malignant():
    _, y = load_data()
    assert y.name == "malignant"


def test_remapped_positive_class_count_matches_malignant():
    _, y = load_data()
    assert int(y.sum()) == 212
    assert int((y == 0).sum()) == 357


def test_feature_shape():
    X, y = load_data()
    assert X.shape == (569, 30)
    assert y.shape == (569,)
