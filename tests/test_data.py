
import pandas as pd
import pytest

from src.HousePrice.data import (
    validate_dataset,
    split_features_target,
    validate_features,
)


def make_sample_data():
    return pd.DataFrame({
        "Order": [1, 2, 3],
        "PID": [1001, 1002, 1003],
        "GrLivArea": [1200, 1500, 1800],
        "Neighborhood": ["A", "B", "A"],
        "SalePrice": [150000, 200000, 250000],
    })


def test_valid_dataset():
    df = make_sample_data()

    validate_dataset(df)


def test_missing_target():
    df = make_sample_data()
    df = df.drop(columns=["SalePrice"])

    with pytest.raises(ValueError):
        validate_dataset(df)


def test_missing_target_value():
    df = make_sample_data()
    df.loc[0, "SalePrice"] = None

    with pytest.raises(ValueError):
        validate_dataset(df)


def test_split_features_target():
    df = make_sample_data()

    X, y = split_features_target(df)

    assert "SalePrice" not in X.columns
    assert "Order" not in X.columns
    assert "PID" not in X.columns
    assert len(X) == len(y)


def test_feature_validation():
    X = pd.DataFrame({
        "GrLivArea": [1200],
        "Neighborhood": ["A"],
    })

    expected = ["Neighborhood", "GrLivArea"]

    result = validate_features(
        X,
        expected_columns=expected,
    )

    assert result.columns.tolist() == expected


def test_feature_validation_rejects_missing_column():
    X = pd.DataFrame({
        "GrLivArea": [1200],
    })

    with pytest.raises(ValueError):
        validate_features(
            X,
            expected_columns=["GrLivArea", "Neighborhood"],
        )
