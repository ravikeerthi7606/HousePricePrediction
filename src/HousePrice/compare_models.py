
import logging

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from HousePrice.config_loader import load_config
from HousePrice.data import load_dataset, split_features_target
from HousePrice.preprocessing import build_preprocessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    df = load_dataset()
    X, y = split_features_target(df)

    seed = config["project"]["random_state"]
    test_size = config["training"].get("test_size", 0.15)
    validation_size = config["training"].get("validation_size", 0.15)

    # Reproduce the existing training split.
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=test_size + validation_size,
        random_state=seed,
    )

    X_val, _, y_val, _ = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_size / (test_size + validation_size),
        random_state=seed,
    )

    preprocessor = build_preprocessor(X_train)

    candidates = {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            learning_rate=0.1,
            subsample=1.0,
            random_state=seed,
        ),
        "Lasso": Lasso(
            alpha=1.0,
            max_iter=20000,
        ),
    }

    # Use only the training partition for CV.
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    results = []

    for name, estimator in candidates.items():
        logger.info("Evaluating %s", name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "r2": "r2",
            },
            n_jobs=1,
        )

        # Train on the full training partition.
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_val)

        results.append({
            "model": name,
            "cv_rmse_mean": -scores["test_rmse"].mean(),
            "cv_rmse_std": scores["test_rmse"].std(),
            "cv_mae_mean": -scores["test_mae"].mean(),
            "cv_r2_mean": scores["test_r2"].mean(),
            "val_rmse": np.sqrt(
                mean_squared_error(y_val, predictions)
            ),
            "val_mae": mean_absolute_error(y_val, predictions),
            "val_r2": r2_score(y_val, predictions),
        })

    result_df = pd.DataFrame(results)

    print("\nCandidate comparison:")
    print(
        result_df.sort_values("val_rmse").to_string(
            index=False,
            float_format=lambda x: f"{x:,.4f}",
        )
    )

    output_path = "model_comparison.csv"
    result_df.to_csv(output_path, index=False)

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
