
import json
from pydantic import ConfigDict, create_model

from HousePrice.config_loader import PROJECT_ROOT

METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"


def load_feature_schema():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    schema = metadata.get("feature_schema")
    columns = metadata.get("feature_columns")

    if not isinstance(schema, dict) or not isinstance(columns, list):
        raise ValueError("Invalid feature schema metadata.")

    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate feature names in metadata.")

    if set(columns) != set(schema):
        raise ValueError("Feature schema and column list mismatch.")

    for column in columns:
        if schema[column].get("type") not in {
            "numeric",
            "categorical",
        }:
            raise ValueError(f"Unsupported type for {column}")

    return columns, schema


def build_request_model():
    columns, schema = load_feature_schema()

    fields = {}

    for column in columns:
        feature_type = schema[column]["type"]

        if feature_type == "numeric":
            field_type = float | None
        else:
            field_type = str | None

        fields[column] = (field_type, ...)

    return create_model(
        "HousePriceRequest",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


HousePriceRequest = build_request_model()
