
from pathlib import Path

import numpy as np
import pandas as pd


from config_loader import (
    load_config,
    PROJECT_ROOT,
)


def load_dataset(path=None):
    """Load the house price dataset."""

    config = load_config()

    if path is None:
        path = PROJECT_ROOT / config["data"]["path"]
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError("Expected a CSV file.")

    df = pd.read_csv(path)

    validate_dataset(df)

    return df


def validate_dataset(df):
    """Validate the raw dataset."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names found.")

    config = load_config()
    target = config["data"]["target"]

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' is missing."
        )

    if df[target].isna().any():
        raise ValueError(
            f"Target '{target}' contains missing values."
        )

    if not pd.api.types.is_numeric_dtype(df[target]):
        raise TypeError("Target must be numeric.")

    target_values = df[target].to_numpy(dtype=float)

    if not np.isfinite(target_values).all():
        raise ValueError(
            "Target contains infinite or invalid values."
        )

    if (target_values <= 0).any():
        raise ValueError(
            "SalePrice must be positive."
        )


def split_features_target(df):
    """Separate features and target."""

    validate_dataset(df)

    config = load_config()

    target = config["data"]["target"]
    drop_columns = config["data"].get(
        "drop_columns", []
    )

    # Do not silently ignore incorrectly configured columns.
    missing_drop_columns = [
        col for col in drop_columns
        if col not in df.columns
    ]

    if missing_drop_columns:
        raise ValueError(
            f"Configured drop columns are missing: "
            f"{missing_drop_columns}"
        )

    columns_to_drop = [target, *drop_columns]

    X = df.drop(columns=columns_to_drop).copy()
    y = df[target].copy()

    if X.empty:
        raise ValueError("No features remain.")

    return X, y


def validate_features(X, expected_columns=None):
    """Validate feature structure."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "Features must be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError("Feature data is empty.")

    if X.columns.duplicated().any():
        raise ValueError("Duplicate feature names found.")

    if expected_columns is not None:
        missing = set(expected_columns) - set(X.columns)
        extra = set(X.columns) - set(expected_columns)

        if missing or extra:
            raise ValueError(
                f"Feature mismatch. "
                f"Missing: {sorted(missing)}; "
                f"Extra: {sorted(extra)}"
            )

        # Align column order to the training schema.
        X = X.loc[:, expected_columns]

    return X
