import pytest

from churnbench.config import load_settings
from churnbench.data import file_sha256, load_dataset
from churnbench.features import engineer


def test_load_coerces_types(csv_path):
    X, y = load_dataset(csv_path)
    assert X["TotalCharges"].dtype.kind == "f"
    assert X["TotalCharges"].isna().sum() == 0
    assert set(y.unique()) <= {0, 1}
    assert "customerID" not in X.columns


def test_hash_is_stable(csv_path):
    assert file_sha256(csv_path) == file_sha256(csv_path)


def test_engineer_adds_columns(csv_path):
    X, _ = load_dataset(csv_path)
    Xe = engineer(X)
    for col in ["service_count", "avg_monthly_spend", "tenure_band", "on_long_contract"]:
        assert col in Xe.columns
    assert len(Xe) == len(X)


def test_env_override(monkeypatch):
    monkeypatch.setenv("CHURNBENCH_SEEDS", "1,2")
    monkeypatch.setenv("CHURNBENCH_BALANCE_CLASSES", "false")
    cfg = load_settings("configs/default.yaml")
    assert cfg.seeds == [1, 2]
    assert cfg.balance_classes is False


def test_bad_model_rejected(monkeypatch):
    monkeypatch.setenv("CHURNBENCH_MODELS", "magic_model")
    with pytest.raises(ValueError):
        load_settings("configs/default.yaml")
