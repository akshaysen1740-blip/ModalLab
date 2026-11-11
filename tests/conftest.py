import pytest

from churnbench.data import make_synthetic


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "churn.csv"
    make_synthetic(600, seed=1).to_csv(path, index=False)
    return path
