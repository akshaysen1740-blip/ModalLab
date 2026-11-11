# ChurnBench

A small, config-driven lab for comparing churn models **reproducibly**. It trains a few
scikit-learn models over several random seeds, reports how stable each one is, finds a
sensible decision threshold, and logs everything to MLflow. The winner is registered with
an alias so it can be loaded by name later.

Built on the public **Telco Customer Churn** dataset (7,043 customers, ~26.5% churn).

```
YAML config ─┐
             ├─> load + validate + hash ─> (optional) feature engineering
CSV dataset ─┘                                     │
                                                   ▼
             for each model × seed: stratified split ─> sklearn Pipeline ─> metrics
                                                   │        (scale + one-hot + model)
                                                   ▼
             threshold sweep ─> plots ─> MLflow (parent run per model, child run per seed)
                                                   │
                                                   ▼
             leaderboard (mean/std/min/max) ─> register best model, alias "candidate"
```

## What it does

- **Reproducible splits**: fixed seeds, stratified on the target, and a SHA-256 fingerprint of the data file logged with every run.
- **Multi-seed evaluation**: each model runs on every seed; results are summarised as mean / std / min / max so you see stability, not one lucky split.
- **Class-imbalance switch**: `balance_classes: true` turns on `class_weight="balanced"` for all models.
- **Optional feature engineering**: service count, average monthly spend, new-customer flag, long-contract flag, tenure band. Toggle with `engineer_features`.
- **Three models**: Logistic Regression, Random Forest, HistGradientBoosting (no XGBoost dependency needed).
- **Threshold sweep**: precision / recall / F1 from 0.10 to 0.90, logged as a table and a plot; the best-F1 threshold is reported per run.
- **MLflow tracking**: params, metrics, confusion matrix and threshold plots, nested runs, model registry with an alias and a `suggested_threshold` tag.
- **Prediction script**: loads `churnbench-best@candidate` and scores customers with the suggested threshold.

## Layout

```
churnbench/
├── configs/
│   ├── default.yaml        # engineered features + balanced classes, registers best model
│   └── raw.yaml            # baseline: raw features, no balancing
├── data/                   # dataset goes here (git-ignored)
├── scripts/
│   ├── get_data.py         # download real data, or --synthetic for offline fake data
│   └── predict.py          # score customers with the registered model
├── src/churnbench/
│   ├── config.py           # dataclass settings: YAML + CHURNBENCH_* env overrides
│   ├── data.py             # loading, schema check, type coercion, hashing, synthetic data
│   ├── features.py         # optional feature engineering
│   ├── models.py           # model factory + sklearn Pipeline
│   ├── evaluate.py         # metrics, threshold sweep, plots, multi-seed summary
│   └── pipeline.py         # orchestrator + model registration
├── tests/                  # unit + end-to-end smoke test (synthetic data, no network)
├── .github/workflows/ci.yml
├── pyproject.toml
└── requirements.txt
```

## Quickstart

```bash
# 1. Virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / macOS

# 2. Install
pip install -r requirements.txt
pip install -e .

# 3. Get data (real)  -- or add --synthetic to work offline
python scripts/get_data.py

# 4. Run the baseline, then the full config
python -m churnbench --config configs/raw.yaml
python -m churnbench --config configs/default.yaml

# 5. (Optional) Browse results in MLflow -- open http://127.0.0.1:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 6. Score customers with the registered model
python scripts/predict.py --top 10

# 7. Tests
pytest -v
```

A leaderboard is printed after each run and saved to `results/<experiment>_summary.csv`.

## Configuration

Every setting lives in a YAML file under `configs/`. Any of them can be overridden with an
environment variable named `CHURNBENCH_<SETTING>` (lists are comma-separated):

```bash
# Linux / macOS
CHURNBENCH_SEEDS=1,2,3,4,5 CHURNBENCH_MODELS=logreg,hist_gb python -m churnbench

# Windows (cmd)
set CHURNBENCH_EXPERIMENT_NAME=churn-v2
python -m churnbench
```

| Key | Meaning |
|---|---|
| `experiment_name` | MLflow experiment |
| `tracking_uri` | MLflow store (default `sqlite:///mlflow.db`) |
| `data_path` | CSV location |
| `seeds` | seeds to evaluate each model on |
| `test_size` | hold-out fraction |
| `engineer_features` | add engineered columns |
| `balance_classes` | use `class_weight="balanced"` |
| `models` | any of `logreg`, `random_forest`, `hist_gb` |
| `register_best` | register the top model by mean F1 |
| `registered_model_name` | name in the model registry |

## Results

Real Telco data, 3 seeds (7, 42, 2024), 80/20 stratified split, F1 at threshold 0.5.

**Baseline** (`raw.yaml`: raw features, no balancing)

| Model | F1 (mean ± std) | Recall (mean) | ROC AUC |
|---|---|---|---|
| Logistic Regression | 0.593 ± 0.024 | 0.545 | 0.847 |
| Random Forest | 0.589 ± 0.026 | 0.517 | 0.851 |
| HistGradientBoosting | 0.583 ± 0.035 | 0.530 | 0.839 |

**Full** (`default.yaml`: engineered features + balanced classes)

| Model | F1 (mean ± std) | Recall (mean) | ROC AUC |
|---|---|---|---|
| Random Forest | 0.643 ± 0.010 | 0.808 | 0.851 |
| Logistic Regression | 0.638 ± 0.017 | 0.818 | 0.851 |
| HistGradientBoosting | 0.626 ± 0.022 | 0.749 | 0.842 |

**Takeaways**

- Balancing the classes is what moves the needle: about **+5 F1 points and +27 recall points**. ROC AUC barely changes, which shows the models' ranking ability was already fine; the gain comes from a better operating point.
- In the baseline, the threshold sweep picks ~0.28 as the best-F1 threshold. That's the same problem seen from another angle: an unbalanced model needs a lower cut-off to catch churners.
- Random Forest and Logistic Regression are within noise of each other (gap smaller than one std). Logistic Regression is simpler to explain and has slightly higher recall, so it's a reasonable pick when interpretability matters; the code registers whichever has the higher mean F1.

Exact numbers can shift slightly between library versions.

## Tech stack

| Part | Tool |
|---|---|
| Language | Python 3.10+ |
| ML | scikit-learn |
| Tracking & registry | MLflow (SQLite backend) |
| Config | dataclasses + YAML |
| Plots | matplotlib |
| CI | GitHub Actions |

## License

MIT
