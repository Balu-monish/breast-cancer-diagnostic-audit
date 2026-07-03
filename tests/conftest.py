import pytest

from bca.data import load_data
from bca.validation import split_data


@pytest.fixture(scope="session")
def full_data():
    return load_data()


@pytest.fixture(scope="session")
def split(full_data):
    X, y = full_data
    return split_data(X, y)
