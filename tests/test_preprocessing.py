
import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from src.HousePrice.preprocessing import build_preprocessor


def make_data():
    return pd.DataFrame({
        "area": [1200, 1500, np.nan, 1800],
        "quality": [5, 7, 6, 8],
        "neighborhood": ["A", "B", "A", "C"],
    })


def test_preprocessor_builds():
    X = make_data()

    preprocessor = build_preprocessor(X)

    assert preprocessor is not None


def test_preprocessor_transforms_data():
    X = make_data()

    preprocessor = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 0
    assert np.isfinite(transformed.data if hasattr(
        transformed, "data"
    ) else transformed).all()


def test_preprocessor_handles_unseen_categories():
    X_train = make_data()

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    X_new = pd.DataFrame({
        "area": [1400],
        "quality": [6],
        "neighborhood": ["UNKNOWN_NEIGHBORHOOD"],
    })

    transformed = preprocessor.transform(X_new)

    assert transformed.shape[0] == 1


def test_preprocessor_handles_missing_categories():
    X = make_data()
    X.loc[0, "neighborhood"] = np.nan

    preprocessor = build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)


def test_empty_data_raises_error():
    X = pd.DataFrame()

    with pytest.raises(ValueError):
        build_preprocessor(X)


def test_duplicate_columns_raise_error():
    X = pd.DataFrame(
        [[1, 2]],
        columns=["area", "area"],
    )

    with pytest.raises(ValueError):
        build_preprocessor(X)
