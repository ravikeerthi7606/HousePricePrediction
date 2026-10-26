
import json
import pandas as pd
from fastapi.testclient import TestClient

from HousePrice.data import load_dataset, split_features_target
from HousePrice.api import app

df = load_dataset()
X, _ = split_features_target(df)

payload = X.iloc[0].to_dict()

# Convert NumPy/Pandas values into JSON-compatible values
payload = json.loads(
    pd.Series(payload).to_json()
)

with TestClient(app) as client:
    health = client.get("/health")
    print("Health:", health.json())

    response = client.post("/predict", json=payload)
    print("Status:", response.status_code)
    print("Prediction:", response.json())

    response.raise_for_status()
