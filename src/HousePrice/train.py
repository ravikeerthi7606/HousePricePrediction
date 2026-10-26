
import json
import logging

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from HousePrice.config_loader import (
    load_config,
    PROJECT_ROOT,
)
from src.HousePrice.data import (
    load_dataset,
    split_features_target,
)
from src.HousePrice.preprocessing import build_preprocessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_metrics(y_true, predictions):
    """Calculate regression evaluation metrics."""

    return {
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, predictions))
        ),
        "mae": float(
            mean_absolute_error(y_true, predictions)
        ),
        "r2": float(
            r2_score(y_true, predictions)
        ),
    }


def main():
    config = load_config()

    # 1. Load data
    logger.info("Loading dataset...")

    df = load_dataset()
    X, y = split_features_target(df)

    logger.info(
        "Dataset loaded: %d rows, %d features",
        X.shape[0],
        X.shape[1],
    )

    # 2. Split data: 70 / 15 / 15
    seed = config["project"]["random_state"]

    test_size = config["training"].get(
        "test_size", 0.15
    )
    validation_size = config["training"].get(
        "validation_size", 0.15
    )

    if not (
        0 < test_size < 1
        and 0 < validation_size < 1
        and test_size + validation_size < 1
    ):
        raise ValueError("Invalid split configuration.")

    X_train, X_holdout, y_train, y_holdout = (
        train_test_split(
            X,
            y,
            test_size=test_size + validation_size,
            random_state=seed,
        )
    )

    # Divide holdout into validation and test
    test_fraction_of_holdout = (
        test_size / (test_size + validation_size)
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_fraction_of_holdout,
        random_state=seed,
    )

    logger.info(
        "Split sizes | train=%d | val=%d | test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # 3. Build model
    model_config = config["model"]

    if model_config["name"] != "GradientBoostingRegressor":
        raise ValueError(
            f"Unsupported model: {model_config['name']}"
        )

    model_params = dict(
        model_config.get("params", {})
    )

    # Prevent unexpected configuration overrides
    model_params["random_state"] = seed

    model = GradientBoostingRegressor(
        **model_params
    )

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # 4. Configure artifact path
    # model_dir = PROJECT_ROOT / "models"
    # model_dir.mkdir(parents=True, exist_ok=True)

    # model_path = model_dir / "house_price_pipeline.joblib"

    # 5. Configure MLflow
    tracking_dir = PROJECT_ROOT / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    # mlflow.set_tracking_uri(
    #     tracking_dir.resolve().as_uri()
    # )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(
        config["project"].get(
            "name", "HousePricePrediction"
        )
    )

    # 6. Train, evaluate, log, and save
    with mlflow.start_run(
        run_name="gradient_boosting_baseline"
    ) as run:

        logger.info("Training model...")

        pipeline.fit(X_train, y_train)

        # Validation predictions
        val_predictions = pipeline.predict(X_val)

        val_metrics = calculate_metrics(
            y_val,
            val_predictions,
        )

        run_id = run.info.run_id

        model_dir = PROJECT_ROOT / "models" / run_id
        model_dir.mkdir(parents=True, exist_ok=False)

        model_path = model_dir / "pipeline.joblib"
        metadata_path = model_dir / "metadata.json"


        logger.info("Validation metrics: %s", val_metrics)

        # 7. Log experiment information
        mlflow.log_params(model_params)

        mlflow.log_param("model_name", model_config["name"])
        mlflow.log_param("random_state", seed)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_val))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("feature_count", X_train.shape[1])

        mlflow.log_metrics({
            f"validation_{key}": value
            for key, value in val_metrics.items()
        })

        mlflow.set_tag("stage", "baseline")
        mlflow.set_tag(
            "selection_metric",
            "validation_rmse",
        )

        # 8. Save complete fitted pipeline
        joblib.dump(pipeline, model_path)

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model_files",
        )

        # Record basic training metadata     
        feature_schema = {}

        for column in X_train.columns:
            dtype = X_train[column].dtype

            if pd.api.types.is_numeric_dtype(dtype):
                feature_type = "numeric"
            elif pd.api.types.is_object_dtype(dtype) or (
                pd.api.types.is_string_dtype(dtype)
                and not pd.api.types.is_numeric_dtype(dtype)
            ) or pd.api.types.is_categorical_dtype(dtype) or (
                pd.api.types.is_bool_dtype(dtype)
            ):
                feature_type = "categorical"
            else:
                raise TypeError(
                    f"Unsupported feature type: {column} ({dtype})"
                )

            feature_schema[column] = {
                "type": feature_type,
                "dtype": str(dtype),
                "nullable": True,
            }

        metadata = {
            "run_id": run_id,
            "model_name": model_config["name"],
            "random_state": seed,
            "feature_columns": X_train.columns.tolist(),
            "feature_schema": feature_schema,
            "target_column": config["data"]["target"],
            "validation_metrics": val_metrics,
            "split_sizes": {
                "train": len(X_train),
                "validation": len(X_val),
                "test": len(X_test),
            },
        }


        metadata_path = model_dir / "metadata.json"

        with metadata_path.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=4)

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model_files",
        )

        mlflow.log_artifact(
            str(metadata_path),
            artifact_path="model_files",
        )

        logger.info("Training completed.")
        logger.info("MLflow run ID: %s", run.info.run_id)
        logger.info("Saved model: %s", model_path)
        logger.info("Validation metrics: %s", val_metrics)


if __name__ == "__main__":
    main()
