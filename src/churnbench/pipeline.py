"""Orchestrator: load -> (engineer) -> for each model & seed: split, fit, evaluate, log."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from churnbench.config import Settings, load_settings
from churnbench.data import file_sha256, load_dataset
from churnbench.evaluate import (compute_metrics, confusion_plot, summarize,
                                 sweep_plot, threshold_sweep)
from churnbench.features import engineer, split_columns
from churnbench.models import build_pipeline, make_model


def run_one(X, y, model_name: str, seed: int, cfg: Settings) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=seed)
    numeric, categorical = split_columns(X)
    pipe = build_pipeline(make_model(model_name, seed, cfg.balance_classes), numeric, categorical)
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]

    metrics = compute_metrics(y_te, proba)
    sweep = threshold_sweep(y_te, proba)
    best = sweep.loc[sweep["f1"].idxmax()]
    metrics.update(best_threshold=float(best["threshold"]), best_threshold_f1=float(best["f1"]))

    with mlflow.start_run(run_name=f"{model_name}-seed{seed}", nested=True):
        mlflow.log_params({"model": model_name, "seed": seed})
        mlflow.log_metrics(metrics)
        mlflow.log_figure(confusion_plot(y_te, proba, f"{model_name} / {seed}"),
                          "confusion_matrix.png")
        mlflow.log_figure(sweep_plot(sweep, f"{model_name} / {seed}"), "threshold_sweep.png")
        mlflow.log_table(sweep, "threshold_sweep.json")
        plt.close("all")
    return {"model": model_name, "seed": seed, "pipeline": pipe, "X_sample": X_te.head(5), **metrics}


def run(cfg: Settings) -> pd.DataFrame:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    X, y = load_dataset(cfg.data_path)
    if cfg.engineer_features:
        X = engineer(X)

    results = []
    for model_name in cfg.models:
        with mlflow.start_run(run_name=model_name) as parent:
            mlflow.log_params({
                "model": model_name, "seeds": ",".join(map(str, cfg.seeds)),
                "test_size": cfg.test_size, "engineer_features": cfg.engineer_features,
                "balance_classes": cfg.balance_classes, "n_rows": len(X),
                "data_sha256": file_sha256(cfg.data_path)[:16],
            })
            rows = [run_one(X, y, model_name, s, cfg) for s in cfg.seeds]
            per_model = pd.DataFrame(rows).drop(columns=["pipeline", "X_sample"])
            summary = summarize(per_model).iloc[0].to_dict()
            mlflow.log_metrics(summary)
            for r in rows:
                r["parent_run_id"] = parent.info.run_id
            results.extend(rows)

    per_seed = pd.DataFrame(results)
    table = summarize(per_seed.drop(columns=["pipeline", "X_sample", "parent_run_id"]))

    out = Path("results"); out.mkdir(exist_ok=True)
    table.to_csv(out / f"{cfg.experiment_name}_summary.csv")
    print("\n=== Leaderboard (sorted by mean F1) ===")
    print(table[["f1_mean", "f1_std", "recall_mean", "roc_auc_mean", "best_threshold_mean"]]
          .to_string())

    if cfg.register_best:
        register_best(per_seed, table, cfg)
    return table


def register_best(per_seed: pd.DataFrame, table: pd.DataFrame, cfg: Settings) -> None:
    """Re-log the winning model (median seed) and register it with alias 'candidate'."""
    winner = table.index[0]
    rows = per_seed[per_seed["model"] == winner].sort_values("f1").reset_index(drop=True)
    pick = rows.iloc[len(rows) // 2]
    pipe, sample = pick["pipeline"], pick["X_sample"]

    with mlflow.start_run(run_id=pick["parent_run_id"]):
        info = mlflow.sklearn.log_model(
            pipe, name="model", input_example=sample,
            serialization_format="cloudpickle",  # we trust our own model; works on MLflow 2 & 3
            signature=infer_signature(sample, pipe.predict_proba(sample)[:, 1]),
            registered_model_name=cfg.registered_model_name)

    client = mlflow.MlflowClient()
    version = info.registered_model_version
    client.set_registered_model_alias(cfg.registered_model_name, "candidate", version)
    client.set_model_version_tag(cfg.registered_model_name, version,
                                 "suggested_threshold", f"{pick['best_threshold']:.2f}")
    print(f"\nRegistered '{winner}' as {cfg.registered_model_name} v{version} "
          f"(alias: candidate, suggested threshold {pick['best_threshold']:.2f})")


def main() -> None:
    p = argparse.ArgumentParser(prog="churnbench")
    p.add_argument("--config", default="configs/default.yaml")
    run(load_settings(p.parse_args().config))


if __name__ == "__main__":
    main()
