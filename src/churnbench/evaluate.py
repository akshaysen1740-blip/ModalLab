"""Metrics, threshold sweep and small plots."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def threshold_sweep(y_true, proba, step: float = 0.05) -> pd.DataFrame:
    rows = []
    for t in np.arange(0.1, 0.91, step):
        m = compute_metrics(y_true, proba, t)
        rows.append({"threshold": round(float(t), 2), **{k: m[k] for k in
                     ("precision", "recall", "f1")}})
    return pd.DataFrame(rows)


def confusion_plot(y_true, proba, title: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, (proba >= 0.5).astype(int),
                                            ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def sweep_plot(sweep: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for col in ("precision", "recall", "f1"):
        ax.plot(sweep["threshold"], sweep[col], label=col)
    ax.set_xlabel("threshold"); ax.set_title(title); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    """mean/std/min/max of each metric per model across seeds."""
    metrics = ["f1", "recall", "precision", "roc_auc", "best_threshold"]
    out = per_seed.groupby("model")[metrics].agg(["mean", "std", "min", "max"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.sort_values("f1_mean", ascending=False).round(4)
