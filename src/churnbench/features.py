"""Optional feature engineering. Turned on/off with `engineer_features` in config."""
from __future__ import annotations

import pandas as pd

from churnbench.data import SERVICE_COLS


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    present = [c for c in SERVICE_COLS if c in X.columns]
    X["service_count"] = (X[present] == "Yes").sum(axis=1)
    X["avg_monthly_spend"] = X["TotalCharges"] / X["tenure"].clip(lower=1)
    X["is_new_customer"] = (X["tenure"] <= 6).astype(int)
    X["on_long_contract"] = (X["Contract"] != "Month-to-month").astype(int)
    X["tenure_band"] = pd.cut(X["tenure"], bins=[-1, 12, 24, 48, 100],
                              labels=["0-1y", "1-2y", "2-4y", "4y+"]).astype(str)
    return X


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical
