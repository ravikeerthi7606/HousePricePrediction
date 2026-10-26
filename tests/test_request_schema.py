
import json

from HousePrice.config_loader import PROJECT_ROOT
from HousePrice.request_schema import load_feature_schema


def test_feature_schema_loads():
    columns, schema = load_feature_schema()

    assert len(columns) == 79
    assert len(schema) == 79
    assert columns[0] == "MS SubClass"
    assert schema["MS SubClass"]["type"] == "numeric"
    assert schema["MS Zoning"]["type"] == "categorical"


def test_schema_does_not_require_dataset():
    metadata_path = PROJECT_ROOT / "models" / "metadata.json"

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    assert "feature_schema" in metadata
    assert len(metadata["feature_columns"]) == 79
