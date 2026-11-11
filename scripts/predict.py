"""Score customers with the registered 'candidate' model and its suggested threshold."""
import argparse
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churnbench.data import load_dataset  # noqa: E402
from churnbench.features import engineer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/telco_churn.csv")
    p.add_argument("--model-name", default="churnbench-best")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(args.model_name, "candidate")
    threshold = float(mv.tags.get("suggested_threshold", 0.5))
    model = mlflow.sklearn.load_model(f"models:/{args.model_name}@candidate")

    X, _ = load_dataset(args.data)
    X = engineer(X)  # registered model was trained on engineered features
    proba = model.predict_proba(X)[:, 1]

    flagged = (proba >= threshold).sum()
    print(f"Model v{mv.version}, threshold {threshold:.2f}: "
          f"{flagged}/{len(X)} customers flagged as likely to churn")
    top = X.assign(churn_risk=proba.round(3)).nlargest(args.top, "churn_risk")
    print(top[["tenure", "Contract", "MonthlyCharges", "churn_risk"]].to_string())


if __name__ == "__main__":
    main()
