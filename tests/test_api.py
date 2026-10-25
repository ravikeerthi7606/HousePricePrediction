
import json

import pandas as pd
from fastapi.testclient import TestClient

from HousePrice.api import app
from HousePrice.data import load_dataset, split_features_target


def get_payload():
    df = load_dataset()
    X, _ = split_features_target(df)

    payload = X.iloc[0].to_dict()

    # Convert Pandas/NumPy values to JSON-compatible values
    return json.loads(pd.Series(payload).to_json())


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint():
    payload = get_payload()

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0


def test_missing_feature():
    payload = get_payload()
    payload.pop(next(iter(payload)))

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_extra_feature():
    payload = get_payload()
    payload["unexpected_feature"] = 100

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_wrong_json_structure():
    with TestClient(app) as client:
        response = client.post("/predict", json=[1, 2, 3])

    assert response.status_code == 422
