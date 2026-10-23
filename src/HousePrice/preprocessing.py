
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

#we don't need to use standard scaler because we are using tree based models which are not affected by feature scaling

def build_preprocessor(X):
    """
    Build a preprocessing pipeline from feature data.

    Numeric:
        Median imputation

    Categorical:
        Most frequent imputation
        One-hot encoding
    """

    if X.empty:
        raise ValueError("Cannot preprocess empty data.")

    if X.columns.duplicated().any():
        raise ValueError("Duplicate feature columns found.")

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    unsupported_features = [
        col for col in X.columns
        if col not in numeric_features
        and col not in categorical_features
    ]

    if unsupported_features:
        raise TypeError(
            "Unsupported feature types: "
            f"{unsupported_features}"
        )

    if not numeric_features and not categorical_features:
        raise ValueError("No supported features found.")

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                )
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor
