"""Settings: YAML file first, then environment overrides.

Env vars use the prefix CHURNBENCH_, e.g.
    CHURNBENCH_EXPERIMENT_NAME=my-exp
    CHURNBENCH_SEEDS=1,2,3
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ENV_PREFIX = "CHURNBENCH_"
VALID_MODELS = {"logreg", "random_forest", "hist_gb"}


@dataclass
class Settings:
    experiment_name: str = "churnbench"
    tracking_uri: str = "sqlite:///mlflow.db"
    data_path: str = "data/telco_churn.csv"
    seeds: list[int] = field(default_factory=lambda: [42])
    test_size: float = 0.2
    engineer_features: bool = True
    balance_classes: bool = True
    models: list[str] = field(default_factory=lambda: ["logreg"])
    register_best: bool = False
    registered_model_name: str = "churnbench-best"

    def validate(self) -> "Settings":
        unknown = set(self.models) - VALID_MODELS
        if unknown:
            raise ValueError(f"Unknown models {unknown}; choose from {VALID_MODELS}")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        return self


def _cast(raw: str, current):
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        items = [x.strip() for x in raw.split(",") if x.strip()]
        return [int(x) for x in items] if current and isinstance(current[0], int) else items
    return raw


def load_settings(path: str | Path | None = None) -> Settings:
    values = {}
    if path:
        values = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in fields(Settings)}
    extra = set(values) - known
    if extra:
        raise ValueError(f"Unknown config keys: {extra}")

    settings = Settings(**values)
    for name in known:
        env = os.getenv(ENV_PREFIX + name.upper())
        if env is not None:
            setattr(settings, name, _cast(env, getattr(settings, name)))
    return settings.validate()
