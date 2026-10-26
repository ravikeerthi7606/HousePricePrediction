
import json
import joblib

from pathlib import Path

from HousePrice.config_loader import PROJECT_ROOT


MODELS_DIR = PROJECT_ROOT / "models"


def load_artifacts(run_id: str):
    """Load a pipeline and its matching metadata."""

    if not run_id or Path(run_id).name != run_id:
        raise ValueError("Invalid model version.")

    artifact_dir = MODELS_DIR / run_id

    model_path = artifact_dir / "pipeline.joblib"
    metadata_path = artifact_dir / "metadata.json"

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if metadata.get("run_id") != run_id:
        raise ValueError("Artifact version mismatch.")

    pipeline = joblib.load(model_path)

    if not hasattr(pipeline, "predict"):
        raise TypeError("Invalid prediction artifact.")

    expected_columns = metadata.get("feature_columns")
    schema = metadata.get("feature_schema")

    if not isinstance(expected_columns, list):
        raise ValueError("Invalid feature columns metadata.")

    if not isinstance(schema, dict):
        raise ValueError("Invalid feature schema metadata.")

    if (
        len(expected_columns) != len(set(expected_columns))
        or set(expected_columns) != set(schema)
    ):
        raise ValueError("Inconsistent feature metadata.")

    model_columns = list(pipeline.feature_names_in_)

    if model_columns != expected_columns:
        raise ValueError("Model and metadata feature mismatch.")

    return pipeline, metadata
