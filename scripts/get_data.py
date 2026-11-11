"""Fetch the Telco churn CSV, or generate a synthetic look-alike for offline use."""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churnbench.data import make_synthetic  # noqa: E402

URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
       "master/data/Telco-Customer-Churn.csv")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/telco_churn.csv")
    p.add_argument("--synthetic", action="store_true", help="generate fake data instead")
    p.add_argument("--rows", type=int, default=3000)
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        make_synthetic(args.rows).to_csv(out, index=False)
        print(f"Wrote {args.rows} synthetic rows -> {out}")
    else:
        urllib.request.urlretrieve(URL, out)
        print(f"Downloaded -> {out}")


if __name__ == "__main__":
    main()
