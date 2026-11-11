"""End-to-end: tiny run on synthetic data with a throwaway MLflow store."""
from churnbench.config import Settings
from churnbench.pipeline import run


def test_pipeline_runs(csv_path, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = Settings(
        experiment_name="smoke", tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        data_path=str(csv_path), seeds=[1, 2], models=["logreg", "hist_gb"],
        register_best=True, registered_model_name="smoke-best",
    ).validate()
    table = run(cfg)
    assert set(table.index) == {"logreg", "hist_gb"}
    assert (table["f1_mean"] > 0).all()
    assert (tmp_path / "results" / "smoke_summary.csv").exists()
