
from pathlib import Path

import yaml


# Project root:
# House Price Prediction/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config():
    """Load project configuration from YAML."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping."
        )

    return config


def resolve_project_path(relative_path):
    """Resolve a project-relative path."""
    return PROJECT_ROOT / relative_path
