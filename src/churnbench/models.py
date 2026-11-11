"""Model factory + sklearn Pipeline builder."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_model(name: str, seed: int, balanced: bool):
    cw = "balanced" if balanced else None
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=cw)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                      class_weight=cw, random_state=seed, n_jobs=-1)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                              class_weight=cw, random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(model, numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return Pipeline([("prep", pre), ("model", model)])
