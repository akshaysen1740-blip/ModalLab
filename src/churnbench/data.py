"""Loading, schema checks and a fingerprint so every run knows its exact data."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Churn"
ID_COL = "customerID"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
REQUIRED = [ID_COL, TARGET, *NUMERIC, "Contract", "InternetService", "PaymentMethod"]
SERVICE_COLS = ["PhoneService", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/get_data.py")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    # TotalCharges arrives as text with blanks for brand-new customers
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    y = df[TARGET].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Churn column must contain only 'Yes'/'No'")
    X = df.drop(columns=[TARGET, ID_COL])
    return X, y.astype(int)


def make_synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Telco-shaped fake data with a real (learnable) churn signal. Used by tests/CI."""
    rng = np.random.default_rng(seed)
    yes_no = lambda p: rng.choice(["Yes", "No"], n, p=[p, 1 - p])  # noqa: E731
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.35, .45, .2])
    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(18, 118, n), 2)

    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[.84, .16]),
        "Partner": yes_no(.5),
        "tenure": tenure,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)",
                                     "Credit card (automatic)"], n),
        "PaperlessBilling": yes_no(.6),
        "MonthlyCharges": monthly,
        "TotalCharges": np.round(monthly * tenure, 2).astype(str),
    })
    for col in SERVICE_COLS:
        df[col] = yes_no(.4)
    df.loc[tenure == 0, "TotalCharges"] = " "  # mimic the real dataset's blanks

    logit = (-1.2 + 1.3 * (contract == "Month-to-month") + 0.7 * (internet == "Fiber optic")
             - 0.04 * tenure + 0.01 * monthly - 0.4 * (df["TechSupport"] == "Yes"))
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df
