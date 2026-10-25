
from pathlib import Path

import pandas as pd
import pytest

from src.HousePrice.predict import load_model, predict_price
from src.HousePrice.config_loader import PROJECT_ROOT
from src.HousePrice.data import load_dataset, split_features_target
from src.HousePrice.predict import predict_batch


def test_model_loads():
    model = load_model()
    assert hasattr(model, "predict")


def test_prediction_returns_finite_number():
    model = load_model()

    df = load_dataset()
    X, _ = split_features_target(df)

    sample = X.iloc[0].to_dict()

    prediction = predict_price(sample, model=model)

    assert isinstance(prediction, float)
    assert pd.notna(prediction)
    assert prediction > 0


def test_missing_feature_is_rejected():
    model = load_model()

    df = load_dataset()
    X, _ = split_features_target(df)

    sample = X.iloc[0].to_dict()
    sample.pop(next(iter(sample)))

    with pytest.raises(ValueError, match="Feature mismatch"):
        predict_price(sample, model=model)


def test_extra_feature_is_rejected():
    model = load_model()

    df = load_dataset()
    X, _ = split_features_target(df)

    sample = X.iloc[0].to_dict()
    sample["unexpected_feature"] = 123

    with pytest.raises(ValueError, match="Feature mismatch"):
        predict_price(sample, model=model)


def test_non_dictionary_input_is_rejected():
    with pytest.raises(TypeError):
        predict_price([1, 2, 3])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        predict_price({})


def test_batch_prediction_returns_correct_rows():
    model = load_model()

    df = load_dataset()
    X, _ = split_features_target(df)

    batch = X.iloc[:10].copy()

    result = predict_batch(batch, model=model)

    assert len(result) == 10
    assert list(result.columns) == ["predicted_price"]
    assert result.index.equals(batch.index)
    assert result["predicted_price"].notna().all()


def test_batch_prediction_matches_individual_predictions():
    model = load_model()

    df = load_dataset()
    X, _ = split_features_target(df)

    batch = X.iloc[:5].copy()

    batch_predictions = predict_batch(batch, model)["predicted_price"].to_numpy()

    individual_predictions = [
        predict_price(row.to_dict(), model)
        for _, row in batch.iterrows()
    ]

    assert batch_predictions == pytest.approx(individual_predictions)


def test_empty_batch_is_rejected():
    model = load_model()

    df = load_dataset()
    X, _ = split_features_target(df)

    with pytest.raises(ValueError, match="cannot be empty"):
        predict_batch(X.iloc[:0], model=model)
