
import pytest

from HousePrice.artifacts import load_artifacts


def test_load_existing_artifacts():
    # Configure this to an actual trained run ID.
    import os

    run_id = os.getenv("HOUSE_PRICE_MODEL_VERSION")

    if not run_id:
        pytest.skip("Model version is not configured.")

    model, metadata = load_artifacts(run_id)

    assert hasattr(model, "predict")
    assert metadata["run_id"] == run_id


def test_invalid_model_version():
    with pytest.raises(ValueError):
        load_artifacts("../invalid")
