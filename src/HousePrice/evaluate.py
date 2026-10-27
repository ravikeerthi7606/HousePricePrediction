
import json

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from HousePrice.config_loader import load_config
from HousePrice.data import load_dataset, split_features_target
from HousePrice.preprocessing import build_preprocessor


def main():
    config = load_config()
    df = load_dataset()
    X, y = split_features_target(df)

    seed = config["project"]["random_state"]
    test_size = config["training"].get("test_size", 0.15)
    val_size = config["training"].get("validation_size", 0.15)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=test_size + val_size,
        random_state=seed,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_size / (test_size + val_size),
        random_state=seed,
    )

    params = config["model"]["params"].copy()
    params["random_state"] = seed

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", GradientBoostingRegressor(**params)),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    metrics = {
        "rmse": float(
            np.sqrt(mean_squared_error(y_test, predictions))
        ),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    report = {
        "model": "GradientBoostingRegressor",
        "evaluation_role": "additional_test_diagnostic",
        "metrics": metrics,
        "negative_predictions": int((predictions < 0).sum()),
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
    }

    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    main()
