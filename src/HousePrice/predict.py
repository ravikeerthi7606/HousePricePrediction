
from pathlib import Path

import os
import joblib
import pandas as pd

from HousePrice.config_loader import PROJECT_ROOT
from HousePrice.artifacts import load_artifacts


MODEL_PATH = PROJECT_ROOT / "models" / "house_price_pipeline.joblib"


def load_model():
    run_id = os.getenv("HOUSE_PRICE_MODEL_VERSION")

    if not run_id:
        raise RuntimeError(
            "HOUSE_PRICE_MODEL_VERSION is not configured."
        )

    pipeline, metadata = load_artifacts(run_id)

    return pipeline
# def load_model(model_path=MODEL_PATH):
#     """Load the trained sklearn pipeline."""
#     model_path = Path(model_path)

#     if not model_path.exists():
#         raise FileNotFoundError(
#             f"Model artifact not found: {model_path}"
#         )

#     pipeline = joblib.load(model_path)

#     if not hasattr(pipeline, "predict"):
#         raise TypeError("Loaded artifact is not a prediction model.")

#     return pipeline


def predict_price(input_data, model=None):
    """
    Predict house price from raw feature values.

    input_data: dict containing one house's features.
    model: optional preloaded sklearn pipeline.
    """
    if not isinstance(input_data, dict):
        raise TypeError("input_data must be a dictionary.")

    if not input_data:
        raise ValueError("input_data cannot be empty.")

    if model is None:
        model = load_model()

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Model does not expose expected feature names."
        )

    expected_features = list(model.feature_names_in_)

    missing = set(expected_features) - set(input_data)
    extra = set(input_data) - set(expected_features)

    if missing or extra:
        raise ValueError(
            f"Feature mismatch. "
            f"Missing: {sorted(missing)}; "
            f"Extra: {sorted(extra)}"
        )

    X = pd.DataFrame(
        [input_data],
        columns=expected_features
    )

    prediction = model.predict(X)

    if len(prediction) != 1:
        raise RuntimeError("Expected exactly one prediction.")

    price = float(prediction[0])

    if not pd.notna(price):
        raise ValueError("Model returned an invalid prediction.")

    return price


def predict_batch(input_data: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Predict prices for multiple houses.

    Parameters
    ----------
    input_data : pd.DataFrame
        Raw house features.
    model : optional
        Preloaded sklearn pipeline.

    Returns
    -------
    pd.DataFrame
        Original row indices and predicted prices.
    """
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("input_data must be a pandas DataFrame.")

    if input_data.empty:
        raise ValueError("input_data cannot be empty.")

    if model is None:
        model = load_model()

    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Model does not expose expected feature names.")

    expected_features = list(model.feature_names_in_)

    if input_data.columns.duplicated().any():
        raise ValueError("Duplicate input columns found.")

    missing = set(expected_features) - set(input_data.columns)
    extra = set(input_data.columns) - set(expected_features)

    if missing or extra:
        raise ValueError(
            f"Feature mismatch. "
            f"Missing: {sorted(missing)}; "
            f"Extra: {sorted(extra)}"
        )

    # Align feature order with training schema
    X = input_data.loc[:, expected_features].copy()

    predictions = model.predict(X)

    if len(predictions) != len(X):
        raise RuntimeError("Prediction count mismatch.")

    if not pd.notna(predictions).all():
        raise ValueError("Model returned invalid predictions.")

    if not all(map(lambda x: float("-inf") < x < float("inf"), predictions)):
        raise ValueError("Model returned non-finite predictions.")

    return pd.DataFrame(
        {
            "predicted_price": predictions
        },
        index=input_data.index
    )


if __name__ == "__main__":
    print("Prediction module ready.")
