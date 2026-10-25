
from typing import Any

import pandas as pd
from pydantic import ConfigDict, create_model

from HousePrice.data import load_dataset, split_features_target


def build_request_model():
    df = load_dataset()
    X, _ = split_features_target(df)

    fields = {}

    for column in X.columns:
        dtype = X[column].dtype

        # Allow nulls because the trained pipeline handles missing data.
        if pd.api.types.is_numeric_dtype(dtype):
            fields[column] = (float | None, ...)
        else:
            fields[column] = (str | None, ...)

    return create_model(
        "HousePriceRequest",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


HousePriceRequest = build_request_model()
